#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a ~10K multilingual AEGIS benchmark by combining reusable OLD samples
with NEW quality-filtered samples selected to stratify the OVERALL dataset.

Core design
===========
1. Map OLD source IDs -> canonical NEW root IDs via NEW root `source_ids`.
2. Map OLD `language` -> `nllb_code` via language_metadata.py.
3. Match OLD and NEW by canonical `(root_id, nllb_code)`.
4. Compute NEW-corpus BERTScore/COMET/chrF percentile thresholds.
5. At the selected percentile (default P85), reuse every unique OLD sample
   whose matched NEW translation passes the quality gate.
6. Add only as many NEW samples as needed to reach `--target-total`.
7. Select those NEW samples so the FINAL combined dataset is stratified:
     - category-aware
     - language-aware
     - root-diverse
   rather than stratifying the NEW add-on in isolation.
8. Special handling for Manipulation:
     - use as many quality-passing Manipulation samples as possible
       (subject to total-size and root-diversity constraints);
     - then balance the remaining categories around the already-reused OLD data.

Quality gate
============
COMET supported:
    DROP only if BERTScore AND chrF AND COMET all fail.

COMET unsupported:
    ignore COMET and DROP only if BERTScore AND chrF both fail.

Output
======
The final JSONL uses the NEW translation-row schema for BOTH reused-old and
newly-selected samples, with traceability fields added:
    sample_origin = "old_reuse" | "new_stratified"
    old_source_root_ids = [...]
    category
    selection_percentile
    sampling_strategy

This keeps one consistent schema for downstream response generation.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import math
import random
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Canonical category names / aliases
# ---------------------------------------------------------------------------

CATEGORY_ALIASES = {
    "controlled/regulated substances": "Controlled/Regulated Substances",
    "controlled/regulated substance": "Controlled/Regulated Substances",
    "guns/illegal weapons": "Guns/Illegal Weapons",
    "guns/weapons": "Guns/Illegal Weapons",
    "criminal planning/confessions": "Criminal Planning/Confessions",
    "suicide and self harm": "Suicide and Self Harm",
    "suicide/self-harm": "Suicide and Self Harm",
    "sexual minor": "Sexual Minor",
    "hate/identity hate": "Hate/Identity Hate",
    "pii/privacy": "PII/Privacy",
    "fraud/deception": "Fraud/Deception",
    "illegal activity": "Illegal Activity",
    "immoral/unethical": "Immoral/Unethical",
    "malware": "Malware",
    "manipulation": "Manipulation",
    "harassment": "Harassment",
    "threat": "Threat",
    "violence": "Violence",
    "sexual": "Sexual",
    "profanity": "Profanity",
    "copyright/trademark/plagiarism": "Copyright/Trademark/Plagiarism",
    "high risk gov. decision making": "High Risk Gov. Decision Making",
    "political/misinformation/conspiracy": "Political/Misinformation/Conspiracy",
    "unauthorized advice": "Unauthorized Advice",
}

CATEGORY_FIELD_PRIORITY = [
    "most_severe_category",
    "category",
    "risk_category",
    "risk_categories",
    "core_category",
    "core_categories",
    "fine_grained_category",
    "fine_grained_categories",
    "violated_categories",
]


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

try:
    import orjson

    def loads(line: bytes):
        return orjson.loads(line)

    JSON_BACKEND = "orjson"

except ImportError:

    def loads(line: bytes):
        return json.loads(line)

    JSON_BACKEND = "json"


def iter_jsonl(path: Path):
    with path.open("rb") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = loads(line)
            except Exception as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{line_number}"
                )
            yield row


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if math.isfinite(value) else np.nan


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    return default


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_language(value: Any) -> str:
    return normalize_text(value).casefold()


def normalize_category(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    return CATEGORY_ALIASES.get(text.casefold(), text)


def flatten_category_value(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(flatten_category_value(item))
        return result

    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(flatten_category_value(item))
        return result

    text = normalize_text(value)
    if not text:
        return []

    # Some older exports stored a one-element list or comma-separated string.
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            return flatten_category_value(parsed)
        except Exception:
            pass

    # Do not split canonical names containing "/" or commas unless clearly
    # multiple values are encoded with semicolons.
    if ";" in text:
        return [
            normalize_category(x)
            for x in text.split(";")
            if normalize_text(x)
        ]

    return [normalize_category(text)]


def extract_primary_category(
    row: Dict[str, Any],
    special_category: str = "Manipulation",
) -> str:
    """
    Extract one category for stratification.

    If a multi-label field contains Manipulation, prefer Manipulation so the
    rare category is not hidden behind another label.
    """
    for field in CATEGORY_FIELD_PRIORITY:
        if field not in row:
            continue

        cats = [
            c for c in flatten_category_value(row.get(field))
            if c and c not in {"Needs Caution", "Other"}
        ]

        if not cats:
            continue

        if special_category in cats:
            return special_category

        return cats[0]

    return "Uncategorized"


def percentile_value(values: np.ndarray, percentile: int) -> float:
    try:
        return float(
            np.percentile(values, percentile, method="linear")
        )
    except TypeError:
        return float(
            np.percentile(
                values,
                percentile,
                interpolation="linear",
            )
        )


def stable_sample_id(root_id: str, nllb_code: str) -> str:
    digest = hashlib.sha1(
        f"{root_id}||{nllb_code}".encode("utf-8")
    ).hexdigest()[:16]
    return f"sample_{digest}"


# ---------------------------------------------------------------------------
# Language metadata
# ---------------------------------------------------------------------------

def load_language_metadata_module(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Language metadata file not found:\n{path}"
        )

    spec = importlib.util.spec_from_file_location(
        "benchmark_language_metadata",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import language metadata file: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_language_entries(module) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for name, value in vars(module).items():
        if name.startswith("__") or not isinstance(value, dict):
            continue

        entries = {
            str(k): v
            for k, v in value.items()
            if isinstance(v, dict) and v.get("nllb_code")
        }

        if not entries:
            continue

        for key, entry in entries.items():
            merged.setdefault(key, entry)

    if not merged:
        raise RuntimeError(
            "No language metadata dictionary with `nllb_code` entries found."
        )

    return merged


def build_language_lookup(
    entries: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    exact_key_lookup: Dict[str, str] = {}
    plain_candidates: Dict[str, set] = {}

    for metadata_key, entry in entries.items():
        nllb_code = normalize_text(entry.get("nllb_code"))
        if not nllb_code:
            continue

        exact_key_lookup[
            normalize_language(metadata_key)
        ] = nllb_code

        plain_language = normalize_language(
            entry.get("language")
        )
        if plain_language:
            plain_candidates.setdefault(
                plain_language,
                set(),
            ).add(nllb_code)

    plain_lookup: Dict[str, str] = {}
    ambiguous_plain: Dict[str, List[str]] = {}

    for language, codes in plain_candidates.items():
        if len(codes) == 1:
            plain_lookup[language] = next(iter(codes))
        else:
            ambiguous_plain[language] = sorted(codes)

    lookup = dict(plain_lookup)
    lookup.update(exact_key_lookup)

    return lookup, ambiguous_plain


def resolve_old_language_to_nllb(
    language_value: Any,
    lookup: Dict[str, str],
    ambiguous_plain: Dict[str, List[str]],
) -> Tuple[Optional[str], Optional[str]]:
    normalized = normalize_language(language_value)

    if not normalized:
        return None, "missing_language"

    if normalized in lookup:
        return lookup[normalized], None

    if normalized in ambiguous_plain:
        return None, "ambiguous_language"

    return None, "language_not_in_metadata"


# ---------------------------------------------------------------------------
# NEW canonical roots / source-ID crosswalk / categories
# ---------------------------------------------------------------------------

def load_root_crosswalk_and_categories(
    new_roots_file: Path,
    special_category: str,
):
    """
    Build:
      old/source ID -> canonical NEW root_id
      canonical NEW root_id -> category
    """
    source_to_canonical: Dict[str, str] = {}
    category_by_root: Dict[str, str] = {}

    stats = Counter()

    for row in iter_jsonl(new_roots_file):
        stats["root_rows"] += 1

        canonical_root_id = normalize_text(
            row.get("root_id")
        )

        if not canonical_root_id:
            stats["missing_root_id"] += 1
            continue

        category_by_root[canonical_root_id] = (
            extract_primary_category(
                row,
                special_category=special_category,
            )
        )

        source_ids = row.get("source_ids")

        if isinstance(source_ids, str):
            source_ids = [source_ids]

        if not isinstance(source_ids, list) or not source_ids:
            stats["roots_missing_source_ids"] += 1
            continue

        for source_id in source_ids:
            source_id = normalize_text(source_id)
            if not source_id:
                continue

            previous = source_to_canonical.get(
                source_id
            )

            if previous is None:
                source_to_canonical[
                    source_id
                ] = canonical_root_id
                stats["unique_source_ids"] += 1

            elif previous == canonical_root_id:
                stats["duplicate_same_mapping"] += 1

            else:
                raise ValueError(
                    "A source ID maps to multiple canonical roots:\n"
                    f"  source_id: {source_id}\n"
                    f"  first:     {previous}\n"
                    f"  second:    {canonical_root_id}"
                )

    return (
        source_to_canonical,
        category_by_root,
        dict(stats),
    )


# ---------------------------------------------------------------------------
# OLD corpus mapping
# ---------------------------------------------------------------------------

def read_and_map_old_data(
    old_file: Path,
    source_to_canonical: Dict[str, str],
    language_lookup: Dict[str, str],
    ambiguous_plain: Dict[str, List[str]],
    category_by_root: Dict[str, str],
):
    """
    Map OLD samples into canonical NEW pair space.

    Returns:
      canonical_pair -> set(old_source_ids)
      canonical_pair -> old category fallback
    """
    stats = Counter()

    all_old_source_language = set()
    canonical_to_old_sources: Dict[
        Tuple[str, str], set
    ] = defaultdict(set)

    old_category_fallback: Dict[
        Tuple[str, str], str
    ] = {}

    for row in iter_jsonl(old_file):
        stats["old_rows"] += 1

        old_source_id = normalize_text(
            row.get("root_id") or row.get("id")
        )

        old_language = (
            row.get("language")
            if row.get("language") is not None
            else row.get("translation_language_key")
        )

        old_language_norm = normalize_language(
            old_language
        )

        if not old_source_id:
            stats["missing_old_root_id"] += 1
            continue

        # Denominator/audit identity before mapping.
        raw_identity = (
            old_source_id,
            old_language_norm or "<missing>",
        )

        if raw_identity in all_old_source_language:
            stats["duplicate_old_source_language"] += 1
        else:
            all_old_source_language.add(
                raw_identity
            )

        nllb_code, lang_error = (
            resolve_old_language_to_nllb(
                old_language,
                language_lookup,
                ambiguous_plain,
            )
        )

        if lang_error is not None:
            stats[lang_error] += 1
            continue

        canonical_root_id = (
            source_to_canonical.get(old_source_id)
        )

        if canonical_root_id is None:
            stats["root_not_in_source_ids"] += 1
            continue

        canonical_pair = (
            canonical_root_id,
            nllb_code,
        )

        canonical_to_old_sources[
            canonical_pair
        ].add(old_source_id)

        old_cat = normalize_category(
            row.get("category")
            or row.get("most_severe_category")
        )

        if (
            canonical_pair not in old_category_fallback
            and old_cat
        ):
            old_category_fallback[
                canonical_pair
            ] = old_cat

        stats["mapped_old_rows"] += 1

    stats["unique_old_source_language"] = len(
        all_old_source_language
    )
    stats["unique_mapped_canonical_pairs"] = len(
        canonical_to_old_sources
    )

    return {
        "all_old_source_language": (
            all_old_source_language
        ),
        "canonical_to_old_sources": dict(
            canonical_to_old_sources
        ),
        "old_category_fallback": (
            old_category_fallback
        ),
        "stats": dict(stats),
    }


# ---------------------------------------------------------------------------
# NEW quality data
# ---------------------------------------------------------------------------

def read_new_quality_data(
    files: List[Path],
    category_by_root: Dict[str, str],
    special_category: str,
):
    """
    Deduplicate NEW rows by (root_id, nllb_code), keeping the first row.
    """
    records: Dict[
        Tuple[str, str],
        Dict[str, Any],
    ] = {}

    stats = Counter()

    for file_index, file_path in enumerate(
        files,
        start=1,
    ):
        print(
            f"\n[{file_index}/{len(files)}] "
            f"Reading {file_path.name}"
        )

        file_rows = 0

        for row in iter_jsonl(file_path):
            stats["new_rows"] += 1
            file_rows += 1

            root_id = normalize_text(
                row.get("root_id")
            )
            nllb_code = normalize_text(
                row.get("nllb_code")
            )

            if not root_id or not nllb_code:
                stats["missing_new_pair_key"] += 1
                continue

            pair = (root_id, nllb_code)

            if pair in records:
                stats["duplicate_new_pairs"] += 1
                continue

            bert = safe_float(
                row.get("bertscore_f1")
            )
            comet = safe_float(
                row.get("comet_score")
            )
            chrf = safe_float(
                row.get("chrf_score")
            )

            if "comet_supported" in row:
                comet_supported = parse_bool(
                    row.get("comet_supported"),
                    default=False,
                )
            else:
                comet_supported = bool(
                    np.isfinite(comet)
                )

            category = category_by_root.get(
                root_id
            )

            if not category:
                category = extract_primary_category(
                    row,
                    special_category=special_category,
                )

            if not category:
                category = "Uncategorized"

            enriched = dict(row)
            enriched["_pair"] = pair
            enriched["_bert"] = bert
            enriched["_comet"] = comet
            enriched["_chrf"] = chrf
            enriched["_comet_supported"] = (
                comet_supported
            )
            enriched["_category"] = category

            records[pair] = enriched

            if file_rows % 100_000 == 0:
                print(
                    f"    processed "
                    f"{file_rows:,} rows..."
                )

        print(
            f"    finished: {file_rows:,} rows"
        )

    if not records:
        raise RuntimeError(
            "No valid NEW root-language pairs loaded."
        )

    return records, dict(stats)


# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------

def compute_thresholds(
    records: Dict[Tuple[str, str], Dict[str, Any]],
    percentile: int,
):
    bert = np.asarray(
        [
            r["_bert"]
            for r in records.values()
            if np.isfinite(r["_bert"])
        ],
        dtype=np.float64,
    )

    chrf = np.asarray(
        [
            r["_chrf"]
            for r in records.values()
            if np.isfinite(r["_chrf"])
        ],
        dtype=np.float64,
    )

    comet = np.asarray(
        [
            r["_comet"]
            for r in records.values()
            if (
                r["_comet_supported"]
                and np.isfinite(r["_comet"])
            )
        ],
        dtype=np.float64,
    )

    if not len(bert):
        raise RuntimeError(
            "No finite BERTScore values."
        )
    if not len(chrf):
        raise RuntimeError(
            "No finite chrF values."
        )
    if not len(comet):
        raise RuntimeError(
            "No finite supported COMET values."
        )

    return {
        "bert": percentile_value(
            bert,
            percentile,
        ),
        "comet": percentile_value(
            comet,
            percentile,
        ),
        "chrf": percentile_value(
            chrf,
            percentile,
        ),
    }


def record_passes_quality(
    record: Dict[str, Any],
    thresholds: Dict[str, float],
) -> bool:
    bert = record["_bert"]
    comet = record["_comet"]
    chrf = record["_chrf"]
    comet_supported = (
        record["_comet_supported"]
    )

    # BERT and chrF are required.
    if not (
        np.isfinite(bert)
        and np.isfinite(chrf)
    ):
        return False

    bert_fail = bert < thresholds["bert"]
    chrf_fail = chrf < thresholds["chrf"]

    if comet_supported:
        comet_fail = (
            (not np.isfinite(comet))
            or comet < thresholds["comet"]
        )
        return not (
            bert_fail
            and chrf_fail
            and comet_fail
        )

    # COMET unsupported: ignore COMET.
    return not (
        bert_fail
        and chrf_fail
    )


def build_percentile_table(
    records: Dict[Tuple[str, str], Dict[str, Any]],
    old_info,
    percentile_step: int,
):
    """
    Produce percentile-analysis rows for reporting.
    """
    percentiles = list(
        range(0, 101, percentile_step)
    )

    canonical_to_old_sources = (
        old_info["canonical_to_old_sources"]
    )

    rows = []
    previous_new = None
    previous_old = None

    for pct in percentiles:
        thresholds = compute_thresholds(
            records,
            pct,
        )

        new_pass_pairs = {
            pair
            for pair, record in records.items()
            if record_passes_quality(
                record,
                thresholds,
            )
        }

        old_reuse_pairs = (
            set(
                canonical_to_old_sources.keys()
            )
            & new_pass_pairs
        )

        new_count = len(new_pass_pairs)
        old_count = len(old_reuse_pairs)

        total_old_raw = len(
            old_info[
                "all_old_source_language"
            ]
        )

        old_cov = (
            100.0
            * old_count
            / total_old_raw
            if total_old_raw
            else 0.0
        )

        rows.append({
            "percentile": pct,
            "bert_threshold": (
                thresholds["bert"]
            ),
            "comet_threshold": (
                thresholds["comet"]
            ),
            "chrf_threshold": (
                thresholds["chrf"]
            ),
            "new_count": new_count,
            "new_delta": (
                None
                if previous_new is None
                else new_count - previous_new
            ),
            "old_reuse": old_count,
            "old_coverage_pct": old_cov,
            "old_delta": (
                None
                if previous_old is None
                else old_count - previous_old
            ),
        })

        previous_new = new_count
        previous_old = old_count

    return rows


# ---------------------------------------------------------------------------
# Target allocation
# ---------------------------------------------------------------------------

def waterfill_targets(
    existing: Counter,
    availability: Counter,
    target_total: int,
    locked_categories: Optional[set] = None,
) -> Dict[str, int]:
    """
    Increase final targets one sample at a time, always giving the next slot to
    the currently smallest stratum that still has availability.

    `existing` is what OLD reuse already contributes.
    `availability` is the maximum total attainable OLD+NEW count.
    """
    locked_categories = (
        locked_categories or set()
    )

    targets = Counter(existing)

    categories = sorted(
        set(availability) | set(existing)
    )

    remaining = (
        target_total - sum(targets.values())
    )

    while remaining > 0:
        candidates = [
            c
            for c in categories
            if (
                c not in locked_categories
                and targets[c]
                < availability[c]
            )
        ]

        if not candidates:
            break

        chosen = min(
            candidates,
            key=lambda c: (
                targets[c],
                -(
                    availability[c]
                    - targets[c]
                ),
                c,
            ),
        )

        targets[chosen] += 1
        remaining -= 1

    return dict(targets)


def compute_category_targets(
    old_counts: Counter,
    availability: Counter,
    target_total: int,
    special_category: str,
) -> Dict[str, int]:
    """
    Special category rule:
      1. retain the OLD contribution;
      2. allocate as much of the special category as is available, up to the
         total benchmark size;
      3. water-fill the remaining slots across the other categories.

    This implements the requested "use Manipulation's max available data, then
    stratify the larger categories" behavior.
    """
    targets = Counter(old_counts)

    remaining = (
        target_total - sum(targets.values())
    )

    if remaining <= 0:
        return dict(targets)

    special_available = availability.get(
        special_category,
        targets[special_category],
    )

    special_extra_capacity = max(
        0,
        special_available
        - targets[special_category],
    )

    special_add = min(
        remaining,
        special_extra_capacity,
    )

    targets[special_category] += (
        special_add
    )
    remaining -= special_add

    if remaining <= 0:
        return dict(targets)

    balanced_target_total = (
        sum(targets.values())
        + remaining
    )

    return waterfill_targets(
        existing=targets,
        availability=availability,
        target_total=balanced_target_total,
        locked_categories={
            special_category
        },
    )


# ---------------------------------------------------------------------------
# Stratified NEW selection
# ---------------------------------------------------------------------------

def select_new_rows_for_overall_stratification(
    old_reuse_records: List[Dict[str, Any]],
    candidate_records: List[Dict[str, Any]],
    target_total: int,
    special_category: str,
    seed: int,
    max_new_per_root: int,
    exclude_old_roots_from_new: bool,
):
    """
    Select NEW rows against deficits in the OVERALL (OLD+NEW) distribution.

    Category:
      - Manipulation is saturated first.
      - Remaining category targets are water-filled from OLD baseline counts.

    Language:
      - final language targets are water-filled from OLD baseline counts.

    Root:
      - exact old pairs are already excluded before this function.
      - by default, roots represented in OLD reuse are excluded from NEW.
      - among NEW rows, at most `max_new_per_root` rows per root.

    Selection occurs over category x language cells. At every step, choose the
    cell with the strongest remaining category and language deficit.
    """
    rng = random.Random(seed)

    old_category_counts = Counter(
        r["_category"]
        for r in old_reuse_records
    )
    old_language_counts = Counter(
        normalize_text(
            r.get("nllb_code")
        )
        for r in old_reuse_records
    )
    old_root_counts = Counter(
        normalize_text(
            r.get("root_id")
        )
        for r in old_reuse_records
    )

    old_roots = set(old_root_counts)

    filtered_candidates = []

    for record in candidate_records:
        root_id = normalize_text(
            record.get("root_id")
        )

        if (
            exclude_old_roots_from_new
            and root_id in old_roots
        ):
            continue

        filtered_candidates.append(record)

    # Candidate availability is unique rows at this point.
    available_category_new = Counter(
        r["_category"]
        for r in filtered_candidates
    )
    available_language_new = Counter(
        normalize_text(
            r.get("nllb_code")
        )
        for r in filtered_candidates
    )

    total_category_availability = (
        old_category_counts
        + available_category_new
    )
    total_language_availability = (
        old_language_counts
        + available_language_new
    )

    category_targets = (
        compute_category_targets(
            old_counts=old_category_counts,
            availability=(
                total_category_availability
            ),
            target_total=target_total,
            special_category=special_category,
        )
    )

    language_targets = waterfill_targets(
        existing=old_language_counts,
        availability=(
            total_language_availability
        ),
        target_total=target_total,
    )

    need_new = max(
        0,
        target_total
        - len(old_reuse_records),
    )

    if need_new == 0:
        return (
            [],
            {
                "category_targets": (
                    category_targets
                ),
                "language_targets": (
                    language_targets
                ),
                "candidate_count": len(
                    filtered_candidates
                ),
                "new_needed": 0,
            },
        )

    # Group candidates by category x language.
    cell_lists: Dict[
        Tuple[str, str],
        List[Dict[str, Any]],
    ] = defaultdict(list)

    for record in filtered_candidates:
        cell = (
            record["_category"],
            normalize_text(
                record.get("nllb_code")
            ),
        )
        cell_lists[cell].append(record)

    # Deterministic random order inside each stratum.
    cell_queues = {}

    for cell, rows in cell_lists.items():
        rng.shuffle(rows)
        cell_queues[cell] = deque(rows)

    current_category = Counter(
        old_category_counts
    )
    current_language = Counter(
        old_language_counts
    )

    selected = []
    selected_pairs = set()
    new_root_counts = Counter()

    def pop_valid_from_cell(cell):
        queue = cell_queues.get(cell)

        if not queue:
            return None

        while queue:
            record = queue[0]

            pair = record["_pair"]
            root_id = normalize_text(
                record.get("root_id")
            )

            if pair in selected_pairs:
                queue.popleft()
                continue

            if (
                new_root_counts[root_id]
                >= max_new_per_root
            ):
                queue.popleft()
                continue

            return queue.popleft()

        return None

    def cell_has_candidate(cell):
        queue = cell_queues.get(cell)
        if not queue:
            return False

        # Clean invalid items from the front.
        while queue:
            record = queue[0]
            root_id = normalize_text(
                record.get("root_id")
            )

            if (
                record["_pair"] in selected_pairs
                or new_root_counts[root_id]
                >= max_new_per_root
            ):
                queue.popleft()
                continue

            return True

        return False

    # Primary pass: respect category targets and language deficits.
    while len(selected) < need_new:
        active_cells = []

        for cell in cell_queues:
            category, language = cell

            cat_need = (
                category_targets.get(
                    category,
                    current_category[category],
                )
                - current_category[category]
            )

            if cat_need <= 0:
                continue

            if not cell_has_candidate(cell):
                continue

            lang_target = language_targets.get(
                language,
                current_language[language],
            )
            lang_need = (
                lang_target
                - current_language[language]
            )

            cat_target = max(
                1,
                category_targets.get(
                    category,
                    1,
                ),
            )
            lang_target_den = max(
                1,
                lang_target,
            )

            # Higher tuple wins.
            score = (
                1 if lang_need > 0 else 0,
                (
                    lang_need
                    / lang_target_den
                ),
                (
                    cat_need
                    / cat_target
                ),
                -current_language[language],
                -current_category[category],
            )

            active_cells.append(
                (score, cell)
            )

        if not active_cells:
            break

        _, best_cell = max(
            active_cells,
            key=lambda x: x[0],
        )

        record = pop_valid_from_cell(
            best_cell
        )

        if record is None:
            continue

        category, language = best_cell
        root_id = normalize_text(
            record.get("root_id")
        )

        selected.append(record)
        selected_pairs.add(
            record["_pair"]
        )
        new_root_counts[root_id] += 1
        current_category[category] += 1
        current_language[language] += 1

    # Redistribution pass: if some planned cells cannot fill because of
    # category x language sparsity or root caps, use any remaining candidate,
    # prioritizing the most underrepresented FINAL category/language.
    while len(selected) < need_new:
        active_cells = []

        for cell in cell_queues:
            if not cell_has_candidate(cell):
                continue

            category, language = cell

            score = (
                -current_category[category],
                -current_language[language],
                1
                if category == special_category
                else 0,
            )

            active_cells.append(
                (score, cell)
            )

        if not active_cells:
            break

        _, best_cell = max(
            active_cells,
            key=lambda x: x[0],
        )

        record = pop_valid_from_cell(
            best_cell
        )

        if record is None:
            continue

        category, language = best_cell
        root_id = normalize_text(
            record.get("root_id")
        )

        selected.append(record)
        selected_pairs.add(
            record["_pair"]
        )
        new_root_counts[root_id] += 1
        current_category[category] += 1
        current_language[language] += 1

    diagnostics = {
        "category_targets": (
            category_targets
        ),
        "language_targets": (
            language_targets
        ),
        "candidate_count_before_root_exclusion": len(
            candidate_records
        ),
        "candidate_count_after_root_exclusion": len(
            filtered_candidates
        ),
        "new_needed": need_new,
        "new_selected": len(selected),
        "old_category_counts": dict(
            old_category_counts
        ),
        "old_language_counts": dict(
            old_language_counts
        ),
        "final_category_counts": dict(
            current_category
        ),
        "final_language_counts": dict(
            current_language
        ),
        "new_root_count": len(
            new_root_counts
        ),
    }

    return selected, diagnostics


# ---------------------------------------------------------------------------
# Output conversion
# ---------------------------------------------------------------------------

def clean_record_for_output(
    record: Dict[str, Any],
    origin: str,
    old_source_root_ids: List[str],
    selection_percentile: int,
):
    out = {
        k: v
        for k, v in record.items()
        if not k.startswith("_")
    }

    root_id = normalize_text(
        out.get("root_id")
    )
    nllb_code = normalize_text(
        out.get("nllb_code")
    )

    out["sample_id"] = stable_sample_id(
        root_id,
        nllb_code,
    )
    out["category"] = record[
        "_category"
    ]
    out["sample_origin"] = origin
    out["old_source_root_ids"] = sorted(
        old_source_root_ids
    )
    out["selection_percentile"] = (
        selection_percentile
    )
    out["sampling_strategy"] = (
        "overall_category_language_root_stratified"
    )

    return out


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def fmt_int(value):
    return "—" if value is None else f"{int(value):,}"


def fmt_float(value, decimals=6):
    return "—" if value is None else f"{float(value):.{decimals}f}"


def fmt_pct(value):
    return "—" if value is None else f"{float(value):.3f}%"


def fmt_delta(value):
    return "—" if value is None else f"{int(value):+,}"


def generate_html(
    percentile_rows,
    selection_summary,
    report_output: Path,
):
    css = """
<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    max-width: 1500px;
    margin: 30px auto;
    padding: 0 25px 60px 25px;
    background: #fafafa;
    color: #222;
    line-height: 1.5;
}
.info {
    background: white;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 18px 22px;
    margin: 22px 0;
}
table {
    border-collapse: collapse;
    width: 100%;
    background: white;
}
th, td {
    padding: 9px 11px;
    border-bottom: 1px solid #e8e8e8;
    text-align: right;
}
th {
    background: #eee;
}
td:first-child, th:first-child {
    text-align: left;
}
code {
    background: #eee;
    padding: 2px 5px;
    border-radius: 4px;
}
</style>
"""

    parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='UTF-8'>",
        "<title>Translation Quality + Overall Stratified Sampling</title>",
        css,
        "</head><body>",
        "<h1>Translation Quality + Overall Stratified Sampling</h1>",
        "<div class='info'>",
        "<h2>Final sample</h2>",
        (
            f"<p><strong>Target total:</strong> "
            f"{selection_summary['target_total']:,}</p>"
        ),
        (
            f"<p><strong>Reused OLD:</strong> "
            f"{selection_summary['old_reused']:,}</p>"
        ),
        (
            f"<p><strong>Selected NEW:</strong> "
            f"{selection_summary['new_selected']:,}</p>"
        ),
        (
            f"<p><strong>Final total:</strong> "
            f"{selection_summary['final_total']:,}</p>"
        ),
        (
            f"<p><strong>Selection percentile:</strong> "
            f"P{selection_summary['selection_percentile']}</p>"
        ),
        (
            "<p>OLD samples are represented by their matched NEW translation "
            "rows, so the final JSONL has one consistent schema.</p>"
        ),
        "</div>",
        "<h2>Percentile analysis</h2>",
        "<table>",
        """
<tr>
<th>Percentile</th>
<th>BERT</th>
<th>COMET</th>
<th>chrF</th>
<th>NEW pass</th>
<th>NEW delta</th>
<th>OLD reusable</th>
<th>OLD coverage %</th>
<th>OLD delta</th>
</tr>
""",
    ]

    for row in percentile_rows:
        parts += [
            "<tr>",
            f"<td>P{row['percentile']}</td>",
            f"<td>{fmt_float(row['bert_threshold'])}</td>",
            f"<td>{fmt_float(row['comet_threshold'])}</td>",
            f"<td>{fmt_float(row['chrf_threshold'], 4)}</td>",
            f"<td>{fmt_int(row['new_count'])}</td>",
            f"<td>{fmt_delta(row['new_delta'])}</td>",
            f"<td>{fmt_int(row['old_reuse'])}</td>",
            f"<td>{fmt_pct(row['old_coverage_pct'])}</td>",
            f"<td>{fmt_delta(row['old_delta'])}</td>",
            "</tr>",
        ]

    parts += [
        "</table>",
        "<h2>Final category distribution</h2>",
        "<table>",
        "<tr><th>Category</th><th>Count</th></tr>",
    ]

    for category, count in sorted(
        selection_summary[
            "final_category_counts"
        ].items(),
        key=lambda x: (-x[1], x[0]),
    ):
        parts.append(
            f"<tr><td>{html.escape(category)}</td>"
            f"<td>{count:,}</td></tr>"
        )

    parts += [
        "</table>",
        "<h2>Final language distribution</h2>",
        "<table>",
        "<tr><th>NLLB code</th><th>Count</th></tr>",
    ]

    for language, count in sorted(
        selection_summary[
            "final_language_counts"
        ].items(),
        key=lambda x: (-x[1], x[0]),
    ):
        parts.append(
            f"<tr><td>{html.escape(language)}</td>"
            f"<td>{count:,}</td></tr>"
        )

    parts += [
        "</table>",
        "</body></html>",
    ]

    report_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_output.write_text(
        "\n".join(parts),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a ~10K OLD+NEW multilingual benchmark whose OVERALL "
            "distribution is stratified by category and language with root "
            "diversity."
        )
    )

    # All file paths are CLI arguments.
    parser.add_argument(
        "--new-dir",
        type=Path,
        required=True,
        help="Directory containing NEW quality-scored JSONL shards.",
    )
    parser.add_argument(
        "--new-pattern",
        required=True,
        help="Glob pattern for NEW quality JSONLs.",
    )
    parser.add_argument(
        "--old-file",
        type=Path,
        required=True,
        help="OLD multilingual JSONL.",
    )
    parser.add_argument(
        "--new-roots-file",
        type=Path,
        required=True,
        help="Canonical NEW roots JSONL containing root_id + source_ids.",
    )
    parser.add_argument(
        "--language-metadata-file",
        type=Path,
        required=True,
        help="Path to language_metadata.py.",
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        required=True,
        help="Final combined OLD-reuse + NEW-stratified JSONL.",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        required=True,
        help="JSON audit for matching, quality, and sampling.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        required=True,
        help="HTML percentile + sampling report.",
    )

    # Sampling/quality controls.
    parser.add_argument(
        "--target-total",
        type=int,
        default=10_000,
        help="Desired final OLD+NEW sample count.",
    )
    parser.add_argument(
        "--selection-percentile",
        type=int,
        default=85,
        help="Percentile whose quality thresholds are used for final selection.",
    )
    parser.add_argument(
        "--percentile-step",
        type=int,
        default=5,
        help="Step used in the percentile-analysis report.",
    )
    parser.add_argument(
        "--special-category",
        default="Manipulation",
        help=(
            "Rare category to saturate before balancing larger categories."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--max-new-per-root",
        type=int,
        default=1,
        help=(
            "Maximum NEW selected rows per canonical root_id. "
            "Default 1 maximizes root diversity."
        ),
    )
    parser.add_argument(
        "--exclude-old-roots-from-new",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If enabled (default), NEW add-on samples cannot use a root_id "
            "already represented in OLD reuse. Use "
            "--no-exclude-old-roots-from-new to allow other languages from "
            "the same root."
        ),
    )

    args = parser.parse_args()

    if args.target_total <= 0:
        raise ValueError(
            "--target-total must be positive."
        )

    if not (
        0 <= args.selection_percentile <= 100
    ):
        raise ValueError(
            "--selection-percentile must be in [0, 100]."
        )

    if (
        args.percentile_step <= 0
        or 100 % args.percentile_step != 0
    ):
        raise ValueError(
            "--percentile-step must be a positive divisor of 100."
        )

    if args.max_new_per_root <= 0:
        raise ValueError(
            "--max-new-per-root must be >= 1."
        )

    for path, label in [
        (args.new_dir, "NEW directory"),
        (args.old_file, "OLD file"),
        (args.new_roots_file, "NEW roots file"),
        (
            args.language_metadata_file,
            "language metadata file",
        ),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found:\n{path}"
            )

    new_files = sorted(
        args.new_dir.glob(
            args.new_pattern
        )
    )

    if not new_files:
        raise FileNotFoundError(
            f"No NEW files matching "
            f"{args.new_pattern!r} in "
            f"{args.new_dir}"
        )

    print("\n" + "=" * 80)
    print("OVERALL STRATIFIED OLD + NEW BENCHMARK BUILDER")
    print("=" * 80)
    print(f"JSON backend: {JSON_BACKEND}")
    print(f"NEW files:    {len(new_files)}")
    print(f"Target total: {args.target_total:,}")
    print(
        f"Quality cut:  "
        f"P{args.selection_percentile}"
    )

    # -----------------------------------------------------------------------
    # Language lookup.
    # -----------------------------------------------------------------------
    language_module = (
        load_language_metadata_module(
            args.language_metadata_file
        )
    )
    language_entries = (
        discover_language_entries(
            language_module
        )
    )
    language_lookup, ambiguous_plain = (
        build_language_lookup(
            language_entries
        )
    )

    # -----------------------------------------------------------------------
    # Root crosswalk and category map.
    # -----------------------------------------------------------------------
    (
        source_to_canonical,
        category_by_root,
        root_stats,
    ) = load_root_crosswalk_and_categories(
        args.new_roots_file,
        special_category=(
            args.special_category
        ),
    )

    # -----------------------------------------------------------------------
    # OLD mapping.
    # -----------------------------------------------------------------------
    old_info = read_and_map_old_data(
        old_file=args.old_file,
        source_to_canonical=(
            source_to_canonical
        ),
        language_lookup=language_lookup,
        ambiguous_plain=ambiguous_plain,
        category_by_root=(
            category_by_root
        ),
    )

    # -----------------------------------------------------------------------
    # NEW quality corpus.
    # -----------------------------------------------------------------------
    new_records, new_stats = (
        read_new_quality_data(
            new_files,
            category_by_root=(
                category_by_root
            ),
            special_category=(
                args.special_category
            ),
        )
    )

    # -----------------------------------------------------------------------
    # Percentile report.
    # -----------------------------------------------------------------------
    percentile_rows = (
        build_percentile_table(
            new_records,
            old_info,
            percentile_step=(
                args.percentile_step
            ),
        )
    )

    print()
    header = (
        f"{'Pct':>5} "
        f"{'BERT':>10} "
        f"{'COMET':>10} "
        f"{'chrF':>10} "
        f"{'New N':>12} "
        f"{'New Δ':>12} "
        f"{'Old reuse':>12} "
        f"{'Old cov%':>10} "
        f"{'Reuse Δ':>12}"
    )
    print(header)
    print("-" * len(header))

    for row in percentile_rows:
        print(
            f"{row['percentile']:>5} "
            f"{row['bert_threshold']:>10.6f} "
            f"{row['comet_threshold']:>10.6f} "
            f"{row['chrf_threshold']:>10.4f} "
            f"{row['new_count']:>12,} "
            f"{(0 if row['new_delta'] is None else row['new_delta']):>+12,} "
            f"{row['old_reuse']:>12,} "
            f"{row['old_coverage_pct']:>9.3f}% "
            f"{(0 if row['old_delta'] is None else row['old_delta']):>+12,}"
        )

    # -----------------------------------------------------------------------
    # Apply selected quality percentile.
    # -----------------------------------------------------------------------
    thresholds = compute_thresholds(
        new_records,
        args.selection_percentile,
    )

    quality_pass_pairs = {
        pair
        for pair, record in new_records.items()
        if record_passes_quality(
            record,
            thresholds,
        )
    }

    old_canonical_pairs = set(
        old_info[
            "canonical_to_old_sources"
        ].keys()
    )

    old_reuse_pairs = (
        old_canonical_pairs
        & quality_pass_pairs
    )

    old_reuse_records = []

    for pair in sorted(old_reuse_pairs):
        record = dict(
            new_records[pair]
        )

        # Fallback to OLD category if the canonical root table did not give
        # a useful category.
        if record["_category"] == "Uncategorized":
            fallback = old_info[
                "old_category_fallback"
            ].get(pair)
            if fallback:
                record["_category"] = fallback

        old_reuse_records.append(
            record
        )

    print("\n" + "=" * 80)
    print(
        f"P{args.selection_percentile} "
        f"REUSE + STRATIFIED NEW SELECTION"
    )
    print("=" * 80)
    print(
        f"Quality-passing NEW pairs: "
        f"{len(quality_pass_pairs):,}"
    )
    print(
        f"Reusable OLD canonical pairs: "
        f"{len(old_reuse_records):,}"
    )

    if len(old_reuse_records) > args.target_total:
        raise RuntimeError(
            "Reusable OLD set already exceeds --target-total. "
            "Increase --target-total or add an OLD downsampling policy."
        )

    # Candidate NEW = quality-passing NEW pairs not already reused from OLD.
    candidate_new_records = [
        new_records[pair]
        for pair in quality_pass_pairs
        if pair not in old_reuse_pairs
    ]

    selected_new_records, sampling_diag = (
        select_new_rows_for_overall_stratification(
            old_reuse_records=(
                old_reuse_records
            ),
            candidate_records=(
                candidate_new_records
            ),
            target_total=(
                args.target_total
            ),
            special_category=(
                args.special_category
            ),
            seed=args.seed,
            max_new_per_root=(
                args.max_new_per_root
            ),
            exclude_old_roots_from_new=(
                args.exclude_old_roots_from_new
            ),
        )
    )

    # -----------------------------------------------------------------------
    # Produce final consistent-schema JSONL.
    # -----------------------------------------------------------------------
    final_rows = []

    for record in old_reuse_records:
        pair = record["_pair"]

        final_rows.append(
            clean_record_for_output(
                record=record,
                origin="old_reuse",
                old_source_root_ids=list(
                    old_info[
                        "canonical_to_old_sources"
                    ].get(pair, [])
                ),
                selection_percentile=(
                    args.selection_percentile
                ),
            )
        )

    for record in selected_new_records:
        final_rows.append(
            clean_record_for_output(
                record=record,
                origin="new_stratified",
                old_source_root_ids=[],
                selection_percentile=(
                    args.selection_percentile
                ),
            )
        )

    # Stable order: reused OLD first, then new; within each group by category,
    # language, root ID for easy review.
    final_rows.sort(
        key=lambda r: (
            0
            if r["sample_origin"]
            == "old_reuse"
            else 1,
            r.get("category", ""),
            r.get("nllb_code", ""),
            r.get("root_id", ""),
        )
    )

    write_jsonl(
        final_rows,
        args.sample_output,
    )

    # Final distributions.
    final_category_counts = Counter(
        row.get("category", "Uncategorized")
        for row in final_rows
    )
    final_language_counts = Counter(
        row.get("nllb_code", "")
        for row in final_rows
    )
    final_root_counts = Counter(
        row.get("root_id", "")
        for row in final_rows
    )

    origin_counts = Counter(
        row["sample_origin"]
        for row in final_rows
    )

    selection_summary = {
        "target_total": args.target_total,
        "selection_percentile": (
            args.selection_percentile
        ),
        "thresholds": thresholds,
        "old_reused": origin_counts[
            "old_reuse"
        ],
        "new_selected": origin_counts[
            "new_stratified"
        ],
        "final_total": len(final_rows),
        "unique_final_roots": len(
            final_root_counts
        ),
        "unique_final_languages": len(
            [
                k
                for k in final_language_counts
                if k
            ]
        ),
        "final_category_counts": dict(
            final_category_counts
        ),
        "final_language_counts": dict(
            final_language_counts
        ),
        "max_rows_for_any_root": (
            max(final_root_counts.values())
            if final_root_counts
            else 0
        ),
    }

    audit = {
        "inputs": {
            "new_dir": str(
                args.new_dir
            ),
            "new_pattern": (
                args.new_pattern
            ),
            "old_file": str(
                args.old_file
            ),
            "new_roots_file": str(
                args.new_roots_file
            ),
            "language_metadata_file": str(
                args.language_metadata_file
            ),
        },
        "config": {
            "target_total": (
                args.target_total
            ),
            "selection_percentile": (
                args.selection_percentile
            ),
            "percentile_step": (
                args.percentile_step
            ),
            "special_category": (
                args.special_category
            ),
            "seed": args.seed,
            "max_new_per_root": (
                args.max_new_per_root
            ),
            "exclude_old_roots_from_new": (
                args.exclude_old_roots_from_new
            ),
        },
        "root_crosswalk_stats": (
            root_stats
        ),
        "old_mapping_stats": (
            old_info["stats"]
        ),
        "new_quality_stats": (
            new_stats
        ),
        "sampling_diagnostics": (
            sampling_diag
        ),
        "selection_summary": (
            selection_summary
        ),
        "percentile_analysis": (
            percentile_rows
        ),
    }

    args.audit_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.audit_output.write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    generate_html(
        percentile_rows=(
            percentile_rows
        ),
        selection_summary=(
            selection_summary
        ),
        report_output=(
            args.report_output
        ),
    )

    print("\n" + "=" * 80)
    print("FINAL DATASET")
    print("=" * 80)
    print(
        f"OLD reused:    "
        f"{origin_counts['old_reuse']:,}"
    )
    print(
        f"NEW selected:  "
        f"{origin_counts['new_stratified']:,}"
    )
    print(
        f"Final total:   "
        f"{len(final_rows):,}"
    )
    print(
        f"Unique roots:  "
        f"{len(final_root_counts):,}"
    )
    print(
        f"Languages:     "
        f"{selection_summary['unique_final_languages']:,}"
    )

    if len(final_rows) < args.target_total:
        print(
            "\nWARNING: Could not reach the requested target total "
            "under the current quality/root/category constraints."
        )
        print(
            "Try one or more of:\n"
            "  --no-exclude-old-roots-from-new\n"
            "  --max-new-per-root 2\n"
            "  a lower --selection-percentile"
        )

    print("\nFinal category counts:")
    for category, count in sorted(
        final_category_counts.items(),
        key=lambda x: (-x[1], x[0]),
    ):
        print(
            f"  {category:<40} "
            f"{count:>7,}"
        )

    print(
        f"\nSample JSONL:\n"
        f"{args.sample_output}"
    )
    print(
        f"\nAudit JSON:\n"
        f"{args.audit_output}"
    )
    print(
        f"\nHTML report:\n"
        f"{args.report_output}"
    )


if __name__ == "__main__":
    main()