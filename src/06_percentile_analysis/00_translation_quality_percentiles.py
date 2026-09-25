#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a combined HTML percentile report for multilingual translation quality
and OLD -> NEW data reuse.

OLD -> NEW matching
-------------------
The OLD and NEW corpora use different root-id schemes, so they are NOT joined
directly on root_id.

For each OLD row:

    OLD root_id
        -> look up in NEW root table's source_ids[]
        -> canonical NEW root_id

    OLD language
        -> look up in language_metadata.py
        -> nllb_code

The canonical join key is therefore:

    (canonical_new_root_id, nllb_code)

That key is matched against the NEW translation-quality rows, which already
contain:

    root_id
    nllb_code
    bertscore_f1
    comet_supported
    comet_score
    chrf_score

Percentile thresholds are computed ONLY from the NEW corpus.

Quality gate
------------
For a COMET-supported row:
    DROP only if BERTScore, chrF, and COMET all fail their thresholds.

For a COMET-unsupported row:
    Ignore COMET and DROP only if BOTH BERTScore and chrF fail.

A row must have finite BERTScore and chrF to be eligible for quality analysis.

OLD-data coverage
-----------------
"Old data covered %" uses ALL unique OLD source-root/language pairs as the
denominator. This means source IDs that cannot be mapped to a canonical NEW root,
languages that cannot be mapped to NLLB codes, or canonical pairs absent from
the NEW translation corpus correctly reduce OLD-data coverage.

The OLD corpus's own quality scores are not used for the percentile gate,
because the OLD schema may not contain chrF. Matched OLD rows are evaluated
using the corresponding NEW translation's quality scores.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


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
    """Stream JSON objects from a JSONL file."""
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
    """Conservative normalization for IDs/language lookup."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_language(value: Any) -> str:
    return normalize_text(value).casefold()


def percentile_value(values: np.ndarray, percentile: int) -> float:
    try:
        return float(np.percentile(values, percentile, method="linear"))
    except TypeError:
        # Compatibility with older NumPy.
        return float(
            np.percentile(values, percentile, interpolation="linear")
        )


# ---------------------------------------------------------------------------
# Language metadata loader
# ---------------------------------------------------------------------------

def load_language_metadata_module(path: Path):
    """
    Import language_metadata.py from an arbitrary path.

    The variable containing the metadata dictionary does not need a fixed name.
    We discover dict-valued module globals whose values contain `nllb_code`.
    """
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
    """
    Discover and merge language metadata dictionaries in the imported module.

    Expected entry shape, for example:

        'Acehnese (Arabic)': {
            'nllb_code': 'ace_Arab',
            'language': 'Acehnese',
            ...
        }
    """
    merged: Dict[str, Dict[str, Any]] = {}

    for name, value in vars(module).items():
        if name.startswith("__") or not isinstance(value, dict):
            continue

        valid_entries = 0
        for key, entry in value.items():
            if isinstance(entry, dict) and entry.get("nllb_code"):
                valid_entries += 1

        if valid_entries == 0:
            continue

        for key, entry in value.items():
            if not isinstance(entry, dict):
                continue
            if not entry.get("nllb_code"):
                continue

            # Preserve the first occurrence if several module dictionaries
            # expose the same metadata.
            merged.setdefault(str(key), entry)

    if not merged:
        raise RuntimeError(
            "Could not find a language metadata dictionary containing "
            "entries with an 'nllb_code' field."
        )

    return merged


def build_language_lookup(
    entries: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Build OLD language-name -> NLLB-code lookup.

    Resolution priority is handled later:
      1. exact top-level metadata key
      2. metadata entry['language'], only when that plain language maps
         unambiguously to one NLLB code

    This avoids silently choosing one script for ambiguous language names.
    """
    exact_key_lookup: Dict[str, str] = {}
    plain_candidates: Dict[str, set] = {}

    for metadata_key, entry in entries.items():
        nllb_code = normalize_text(entry.get("nllb_code"))
        if not nllb_code:
            continue

        exact_key_lookup[normalize_language(metadata_key)] = nllb_code

        plain_language = normalize_language(entry.get("language"))
        if plain_language:
            plain_candidates.setdefault(plain_language, set()).add(nllb_code)

    plain_lookup: Dict[str, str] = {}
    ambiguous_plain: Dict[str, List[str]] = {}

    for language, codes in plain_candidates.items():
        if len(codes) == 1:
            plain_lookup[language] = next(iter(codes))
        else:
            ambiguous_plain[language] = sorted(codes)

    # Exact top-level keys are authoritative and can safely override a
    # same-text plain-language key.
    lookup = dict(plain_lookup)
    lookup.update(exact_key_lookup)

    return lookup, ambiguous_plain


def resolve_old_language_to_nllb(
    language_value: Any,
    lookup: Dict[str, str],
    ambiguous_plain: Dict[str, List[str]],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (nllb_code, error_reason).

    Old data primarily uses `language`; callers may pass translation_language_key
    as a fallback.
    """
    normalized = normalize_language(language_value)
    if not normalized:
        return None, "missing_language"

    if normalized in lookup:
        return lookup[normalized], None

    if normalized in ambiguous_plain:
        return None, "ambiguous_language"

    return None, "language_not_in_metadata"


# ---------------------------------------------------------------------------
# NEW canonical root crosswalk
# ---------------------------------------------------------------------------

def build_source_to_canonical_root_map(
    new_roots_file: Path,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Build:

        old/source ID -> canonical NEW root_id

    using each canonical root row's `source_ids` list.
    """
    if not new_roots_file.exists():
        raise FileNotFoundError(
            f"NEW roots file not found:\n{new_roots_file}"
        )

    source_to_canonical: Dict[str, str] = {}

    total_root_rows = 0
    roots_missing_root_id = 0
    roots_missing_source_ids = 0
    source_ids_seen = 0
    duplicate_same_mapping = 0
    conflicting_source_ids = 0

    for row in iter_jsonl(new_roots_file):
        total_root_rows += 1

        canonical_root_id = normalize_text(row.get("root_id"))
        if not canonical_root_id:
            roots_missing_root_id += 1
            continue

        source_ids = row.get("source_ids")

        # Be tolerant of a single string if an older export used one.
        if isinstance(source_ids, str):
            source_ids = [source_ids]

        if not isinstance(source_ids, list) or not source_ids:
            roots_missing_source_ids += 1
            continue

        for source_id in source_ids:
            source_id = normalize_text(source_id)
            if not source_id:
                continue

            source_ids_seen += 1

            previous = source_to_canonical.get(source_id)
            if previous is None:
                source_to_canonical[source_id] = canonical_root_id
            elif previous == canonical_root_id:
                duplicate_same_mapping += 1
            else:
                conflicting_source_ids += 1
                raise ValueError(
                    "A source ID maps to multiple canonical NEW roots:\n"
                    f"  source_id: {source_id}\n"
                    f"  first:     {previous}\n"
                    f"  second:    {canonical_root_id}"
                )

    stats = {
        "total_root_rows": total_root_rows,
        "roots_missing_root_id": roots_missing_root_id,
        "roots_missing_source_ids": roots_missing_source_ids,
        "source_ids_seen": source_ids_seen,
        "unique_source_ids": len(source_to_canonical),
        "duplicate_same_mapping": duplicate_same_mapping,
        "conflicting_source_ids": conflicting_source_ids,
    }

    return source_to_canonical, stats


# ---------------------------------------------------------------------------
# OLD corpus mapping
# ---------------------------------------------------------------------------

def read_and_map_old_data(
    old_file: Path,
    source_to_canonical: Dict[str, str],
    language_lookup: Dict[str, str],
    ambiguous_plain: Dict[str, List[str]],
):
    """
    Convert each OLD source-root/language pair into the canonical NEW key:

        (canonical_new_root_id, nllb_code)

    OLD raw identity is retained separately:

        (old_source_root_id, nllb_code)

    so OLD coverage is measured against the original OLD corpus rather than
    against deduplicated canonical roots.
    """
    print("\n" + "=" * 80)
    print("READING + MAPPING OLD DATA")
    print("=" * 80)

    total_rows = 0
    duplicate_old_raw_pairs = 0

    missing_old_root_id = 0
    root_not_in_new_source_ids = 0

    missing_language = 0
    language_not_in_metadata = 0
    ambiguous_language = 0

    # Every unique old row-language sample for denominator/coverage.
    all_old_raw_keys = set()

    # Only successfully mapped OLD samples:
    #   (old_source_id, nllb_code) -> (canonical_root_id, nllb_code)
    mapped_raw_to_canonical = {}

    old_source_ids = set()
    mapped_canonical_roots = set()

    # Diagnostics by language name.
    unmapped_language_counts: Dict[str, int] = {}

    for row in iter_jsonl(old_file):
        total_rows += 1

        old_source_id = normalize_text(
            row.get("root_id") or row.get("id")
        )

        # Old schema primarily uses "language"; keep the historical
        # translation_language_key as a fallback.
        old_language = (
            row.get("language")
            if row.get("language") is not None
            else row.get("translation_language_key")
        )

        if not old_source_id:
            missing_old_root_id += 1
            continue

        old_source_ids.add(old_source_id)

        # Count the OLD source-language sample in the coverage denominator
        # BEFORE trying to map its language or source ID. This way mapping
        # failures correctly reduce overall OLD-data coverage.
        raw_language_key = normalize_language(old_language) or "<missing>"
        raw_key = (old_source_id, raw_language_key)

        if raw_key in all_old_raw_keys:
            duplicate_old_raw_pairs += 1
            continue

        all_old_raw_keys.add(raw_key)

        nllb_code, lang_error = resolve_old_language_to_nllb(
            old_language,
            language_lookup,
            ambiguous_plain,
        )

        if lang_error is not None:
            lang_name = normalize_text(old_language) or "<missing>"

            if lang_error == "missing_language":
                missing_language += 1
            elif lang_error == "ambiguous_language":
                ambiguous_language += 1
            elif lang_error == "language_not_in_metadata":
                language_not_in_metadata += 1

            unmapped_language_counts[lang_name] = (
                unmapped_language_counts.get(lang_name, 0) + 1
            )
            continue

        canonical_root_id = source_to_canonical.get(old_source_id)
        if canonical_root_id is None:
            root_not_in_new_source_ids += 1
            continue

        canonical_key = (canonical_root_id, nllb_code)
        mapped_raw_to_canonical[raw_key] = canonical_key
        mapped_canonical_roots.add(canonical_root_id)

    print(f"Old rows:                              {total_rows:,}")
    print(f"Unique old source IDs:                 {len(old_source_ids):,}")
    print(f"Unique old source-language pairs:      {len(all_old_raw_keys):,}")
    print(f"Mapped old source-language pairs:      {len(mapped_raw_to_canonical):,}")
    print(f"Mapped canonical NEW roots:            {len(mapped_canonical_roots):,}")
    print(f"Duplicate old source-language pairs:   {duplicate_old_raw_pairs:,}")
    print(f"Missing old root/id:                    {missing_old_root_id:,}")
    print(f"Source ID absent from NEW source_ids:  {root_not_in_new_source_ids:,}")
    print(f"Missing language:                       {missing_language:,}")
    print(f"Language absent from metadata:          {language_not_in_metadata:,}")
    print(f"Ambiguous language names:               {ambiguous_language:,}")

    if unmapped_language_counts:
        print("\nTop unmapped/ambiguous OLD language names:")
        for name, count in sorted(
            unmapped_language_counts.items(),
            key=lambda x: (-x[1], x[0]),
        )[:20]:
            print(f"  {name:<35} {count:>10,}")

    return {
        "total_rows": total_rows,
        "old_source_ids": old_source_ids,
        "all_old_raw_keys": all_old_raw_keys,
        "mapped_raw_to_canonical": mapped_raw_to_canonical,
        "mapped_canonical_roots": mapped_canonical_roots,
        "duplicate_old_raw_pairs": duplicate_old_raw_pairs,
        "missing_old_root_id": missing_old_root_id,
        "root_not_in_new_source_ids": root_not_in_new_source_ids,
        "missing_language": missing_language,
        "language_not_in_metadata": language_not_in_metadata,
        "ambiguous_language": ambiguous_language,
        "unmapped_language_counts": unmapped_language_counts,
    }


# ---------------------------------------------------------------------------
# NEW quality-scored data
# ---------------------------------------------------------------------------

def read_new_quality_data(files: List[Path]):
    """
    Read NEW translation-quality rows and deduplicate by:

        (root_id, nllb_code)

    The first occurrence is retained for analysis. Duplicate pairs are counted
    as diagnostics so they cannot inflate percentile/retention statistics.
    """
    print("\n" + "=" * 80)
    print("READING NEW TRANSLATION-QUALITY DATA")
    print("=" * 80)

    pair_to_record: Dict[
        Tuple[str, str],
        Tuple[float, float, float, bool],
    ] = {}

    total_rows = 0
    missing_pair_key = 0
    duplicate_pairs = 0

    for file_index, file_path in enumerate(files, start=1):
        print(f"\n[{file_index}/{len(files)}] Reading {file_path.name}")
        file_rows = 0

        for row in iter_jsonl(file_path):
            total_rows += 1
            file_rows += 1

            root_id = normalize_text(row.get("root_id"))
            nllb_code = normalize_text(row.get("nllb_code"))

            if not root_id or not nllb_code:
                missing_pair_key += 1
                continue

            pair = (root_id, nllb_code)

            if pair in pair_to_record:
                duplicate_pairs += 1
                continue

            bert = safe_float(row.get("bertscore_f1"))
            chrf = safe_float(row.get("chrf_score"))
            comet = safe_float(row.get("comet_score"))

            # Prefer the explicit support flag. If it is entirely absent,
            # infer support from whether a finite COMET score exists.
            if "comet_supported" in row:
                comet_supported = parse_bool(
                    row.get("comet_supported"),
                    default=False,
                )
            else:
                comet_supported = bool(np.isfinite(comet))

            pair_to_record[pair] = (
                bert,
                comet,
                chrf,
                comet_supported,
            )

            if file_rows % 100_000 == 0:
                print(f"    processed {file_rows:,} rows...")

        print(f"    finished: {file_rows:,} rows")

    if not pair_to_record:
        raise RuntimeError("No valid NEW root-language pairs were loaded.")

    pairs = list(pair_to_record.keys())

    bert = np.asarray(
        [pair_to_record[p][0] for p in pairs],
        dtype=np.float64,
    )
    comet = np.asarray(
        [pair_to_record[p][1] for p in pairs],
        dtype=np.float64,
    )
    chrf = np.asarray(
        [pair_to_record[p][2] for p in pairs],
        dtype=np.float64,
    )
    comet_supported = np.asarray(
        [pair_to_record[p][3] for p in pairs],
        dtype=bool,
    )

    pair_to_index = {
        pair: index
        for index, pair in enumerate(pairs)
    }

    print("\nNEW data summary")
    print(f"Total rows read:                 {total_rows:,}")
    print(f"Unique root-language pairs:      {len(pairs):,}")
    print(f"Duplicate root-language rows:    {duplicate_pairs:,}")
    print(f"Rows missing root/language key:  {missing_pair_key:,}")

    return {
        "pairs": pairs,
        "pair_to_index": pair_to_index,
        "bert": bert,
        "comet": comet,
        "chrf": chrf,
        "comet_supported": comet_supported,
        "total_rows": total_rows,
        "duplicate_pairs": duplicate_pairs,
        "missing_pair_key": missing_pair_key,
    }


# ---------------------------------------------------------------------------
# Percentile + OLD reuse analysis
# ---------------------------------------------------------------------------

def build_old_match_indices(old_info, new_data):
    """
    Match mapped OLD raw samples to NEW canonical translation rows.

    Multiple OLD source IDs may map to the same canonical NEW root after
    deduplication. They remain separate OLD samples for coverage calculations.
    """
    pair_to_index = new_data["pair_to_index"]

    matched_indices = []
    mapped_but_absent_new = 0

    for canonical_pair in old_info["mapped_raw_to_canonical"].values():
        index = pair_to_index.get(canonical_pair)

        if index is None:
            mapped_but_absent_new += 1
            continue

        matched_indices.append(index)

    return (
        np.asarray(matched_indices, dtype=np.int64),
        mapped_but_absent_new,
    )


def compute_quality_pass_mask(
    bert: np.ndarray,
    comet: np.ndarray,
    chrf: np.ndarray,
    comet_supported: np.ndarray,
    bert_threshold: float,
    comet_threshold: float,
    chrf_threshold: float,
):
    """
    Required gate:

    COMET supported:
        fail = BERT fail AND chrF fail AND COMET fail

    COMET unsupported:
        fail = BERT fail AND chrF fail

    A supported-but-missing COMET score is treated as COMET failure.
    """
    eligible = np.isfinite(bert) & np.isfinite(chrf)

    bert_fail = bert < bert_threshold
    chrf_fail = chrf < chrf_threshold

    comet_fail = (
        (~np.isfinite(comet))
        | (comet < comet_threshold)
    )

    fail_supported = (
        comet_supported
        & bert_fail
        & chrf_fail
        & comet_fail
    )

    fail_unsupported = (
        (~comet_supported)
        & bert_fail
        & chrf_fail
    )

    fail = fail_supported | fail_unsupported
    passed = eligible & ~fail

    return passed, eligible


def build_percentile_rows(
    new_data,
    old_info,
    percentile_step: int,
):
    bert = new_data["bert"]
    comet = new_data["comet"]
    chrf = new_data["chrf"]
    comet_supported = new_data["comet_supported"]

    bert_valid = bert[np.isfinite(bert)]
    chrf_valid = chrf[np.isfinite(chrf)]

    # COMET percentile distribution includes only languages explicitly
    # supported by COMET and having a finite score.
    comet_valid = comet[
        comet_supported & np.isfinite(comet)
    ]

    if len(bert_valid) == 0:
        raise RuntimeError("No valid NEW BERTScore values found.")
    if len(chrf_valid) == 0:
        raise RuntimeError("No valid NEW chrF values found.")
    if len(comet_valid) == 0:
        raise RuntimeError(
            "No valid NEW COMET scores found for COMET-supported languages."
        )

    old_match_indices, mapped_but_absent_new = build_old_match_indices(
        old_info,
        new_data,
    )

    total_old_unique_pairs = len(old_info["all_old_raw_keys"])
    exact_old_new_matches = len(old_match_indices)

    percentiles = list(range(0, 101, percentile_step))

    rows = []

    previous_new_count = None
    previous_old_reusable = None

    for percentile in percentiles:
        bert_threshold = percentile_value(
            bert_valid,
            percentile,
        )
        comet_threshold = percentile_value(
            comet_valid,
            percentile,
        )
        chrf_threshold = percentile_value(
            chrf_valid,
            percentile,
        )

        new_pass_mask, new_eligible_mask = compute_quality_pass_mask(
            bert=bert,
            comet=comet,
            chrf=chrf,
            comet_supported=comet_supported,
            bert_threshold=bert_threshold,
            comet_threshold=comet_threshold,
            chrf_threshold=chrf_threshold,
        )

        new_eligible_count = int(
            np.count_nonzero(new_eligible_mask)
        )
        new_count = int(
            np.count_nonzero(new_pass_mask)
        )

        new_retention = (
            100.0 * new_count / new_eligible_count
            if new_eligible_count
            else 0.0
        )

        new_delta = (
            None
            if previous_new_count is None
            else new_count - previous_new_count
        )
        previous_new_count = new_count

        if exact_old_new_matches:
            matched_eligible_mask = new_eligible_mask[
                old_match_indices
            ]
            matched_pass_mask = new_pass_mask[
                old_match_indices
            ]

            matched_old_eligible = int(
                np.count_nonzero(matched_eligible_mask)
            )
            old_reusable = int(
                np.count_nonzero(matched_pass_mask)
            )
        else:
            matched_old_eligible = 0
            old_reusable = 0

        # Denominator intentionally includes ALL unique OLD source-language
        # pairs, so mapping failures and absent NEW translations reduce
        # reusable coverage.
        old_data_coverage_pct = (
            100.0 * old_reusable / total_old_unique_pairs
            if total_old_unique_pairs
            else 0.0
        )

        old_matched_retention_pct = (
            100.0 * old_reusable / matched_old_eligible
            if matched_old_eligible
            else 0.0
        )

        old_reuse_delta = (
            None
            if previous_old_reusable is None
            else old_reusable - previous_old_reusable
        )
        previous_old_reusable = old_reusable

        rows.append({
            "percentile": percentile,
            "bert_threshold": bert_threshold,
            "comet_threshold": comet_threshold,
            "chrf_threshold": chrf_threshold,
            "new_count": new_count,
            "new_retention": new_retention,
            "new_delta": new_delta,
            "old_reusable": old_reusable,
            "old_data_coverage_pct": old_data_coverage_pct,
            "old_reuse_delta": old_reuse_delta,
            "old_matched_retention_pct": old_matched_retention_pct,
        })

    metadata = {
        "json_backend": JSON_BACKEND,

        "new_total_rows": new_data["total_rows"],
        "new_unique_pairs": len(new_data["pairs"]),
        "new_duplicate_pairs": new_data["duplicate_pairs"],
        "new_missing_pair_key": new_data["missing_pair_key"],

        "new_bert_valid": len(bert_valid),
        "new_comet_valid": len(comet_valid),
        "new_chrf_valid": len(chrf_valid),

        "old_total_rows": old_info["total_rows"],
        "old_unique_source_ids": len(old_info["old_source_ids"]),
        "old_unique_raw_pairs": total_old_unique_pairs,
        "old_mapped_pairs": len(
            old_info["mapped_raw_to_canonical"]
        ),
        "old_duplicate_raw_pairs": old_info[
            "duplicate_old_raw_pairs"
        ],

        "old_missing_root_id": old_info["missing_old_root_id"],
        "old_root_not_in_source_ids": old_info[
            "root_not_in_new_source_ids"
        ],
        "old_missing_language": old_info["missing_language"],
        "old_language_not_in_metadata": old_info[
            "language_not_in_metadata"
        ],
        "old_ambiguous_language": old_info[
            "ambiguous_language"
        ],

        "exact_old_new_matches": exact_old_new_matches,
        "mapped_but_absent_new": mapped_but_absent_new,
    }

    return rows, metadata


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def fmt_int(value):
    return "—" if value is None else f"{int(value):,}"


def fmt_float(value, decimals=6):
    if value is None:
        return "—"
    return f"{float(value):.{decimals}f}"


def fmt_pct(value):
    return "—" if value is None else f"{float(value):.3f}%"


def fmt_delta(value):
    return "—" if value is None else f"{int(value):+,}"


def generate_html(
    rows,
    metadata,
    new_files: List[Path],
    old_file: Path,
    new_roots_file: Path,
    language_metadata_file: Path,
    output_file: Path,
    percentile_step: int,
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
h1 { margin-bottom: 5px; }
h2 { margin-top: 24px; }
.subtitle { color: #555; margin-top: 0; }
.info {
    background: white;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 18px 22px;
    margin: 22px 0;
}
.info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 8px 30px;
}
.table-container {
    overflow-x: auto;
    background: white;
    border: 1px solid #ddd;
    border-radius: 8px;
}
table {
    border-collapse: collapse;
    width: 100%;
    min-width: 1200px;
}
th, td {
    padding: 10px 12px;
    border-bottom: 1px solid #e8e8e8;
    text-align: right;
    white-space: nowrap;
}
th {
    position: sticky;
    top: 0;
    background: #eeeeee;
    font-weight: 650;
    z-index: 1;
}
td:first-child, th:first-child { text-align: center; }
tbody tr:hover { background: #f6f6f6; }
.note { font-size: 14px; color: #555; }
code { background: #eee; border-radius: 4px; padding: 2px 5px; }
.delta { font-weight: 600; }
.coverage { font-weight: 700; }
</style>
"""

    parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='UTF-8'>",
        "<title>Translation Quality Percentile and Old-Data Reuse Analysis</title>",
        css,
        "</head><body>",
        "<h1>Translation Quality Percentile and Old-Data Reuse Analysis</h1>",
        (
            "<p class='subtitle'>OLD source IDs are mapped through NEW "
            "<code>source_ids</code>; OLD language names are mapped through "
            "<code>language_metadata.py</code>; final matching uses "
            "<code>(canonical root_id, nllb_code)</code>.</p>"
        ),
        "<div class='info'>",
        "<h2>Dataset summary</h2>",
        "<div class='info-grid'>",
    ]

    summary_items = [
        ("JSON backend", metadata["json_backend"]),
        ("New rows read", f"{metadata['new_total_rows']:,}"),
        ("New unique root-language pairs", f"{metadata['new_unique_pairs']:,}"),
        ("New duplicate pairs ignored", f"{metadata['new_duplicate_pairs']:,}"),
        ("Valid NEW BERT scores", f"{metadata['new_bert_valid']:,}"),
        ("Valid/supported NEW COMET scores", f"{metadata['new_comet_valid']:,}"),
        ("Valid NEW chrF scores", f"{metadata['new_chrf_valid']:,}"),

        ("Old rows read", f"{metadata['old_total_rows']:,}"),
        ("Old unique source IDs", f"{metadata['old_unique_source_ids']:,}"),
        ("Old unique source-language pairs", f"{metadata['old_unique_raw_pairs']:,}"),
        ("Old pairs mapped to canonical root + NLLB", f"{metadata['old_mapped_pairs']:,}"),
        ("Exact OLD -> NEW translation matches", f"{metadata['exact_old_new_matches']:,}"),
        ("Mapped OLD pairs absent from NEW translations", f"{metadata['mapped_but_absent_new']:,}"),
        ("Percentile step", str(percentile_step)),
    ]

    for label, value in summary_items:
        parts.append(
            f"<div><strong>{html.escape(label)}:</strong> "
            f"{html.escape(str(value))}</div>"
        )

    parts += [
        "</div>",
        """
<p class='note'><strong>OLD -> NEW matching:</strong>
OLD <code>root_id</code> (or <code>id</code> fallback) is looked up in the
canonical NEW root table's <code>source_ids</code>. OLD <code>language</code>
is converted to <code>nllb_code</code> using the supplied language metadata.
The resulting <code>(canonical root_id, nllb_code)</code> is joined to the NEW
translation-quality data.</p>

<p class='note'><strong>Quality gate:</strong>
For COMET-supported languages, a row is dropped only when BERTScore, chrF,
and COMET all fail the percentile thresholds. For COMET-unsupported languages,
COMET is ignored and the row is dropped only when both BERTScore and chrF fail.</p>

<p class='note'><strong>Old data covered %:</strong>
retained matched OLD source-language samples divided by all unique OLD
source-language samples. Therefore unresolved source IDs, unresolved languages,
and OLD samples absent from the NEW translation corpus reduce coverage.</p>
""",
        "</div>",
        "<div class='table-container'>",
        "<table>",
        """
<thead>
<tr>
    <th>Percentile</th>
    <th>BERTScore-F1 threshold</th>
    <th>COMET threshold</th>
    <th>chrF threshold</th>
    <th>NEW samples retained</th>
    <th>NEW retention %</th>
    <th>NEW delta</th>
    <th>OLD samples reusable</th>
    <th>OLD data covered %</th>
    <th>OLD reuse delta</th>
    <th>Matched OLD retention %</th>
</tr>
</thead>
<tbody>
""",
    ]

    for row in rows:
        parts += [
            "<tr>",
            f"<td>P{row['percentile']}</td>",
            f"<td>{fmt_float(row['bert_threshold'])}</td>",
            f"<td>{fmt_float(row['comet_threshold'])}</td>",
            f"<td>{fmt_float(row['chrf_threshold'], 4)}</td>",
            f"<td>{fmt_int(row['new_count'])}</td>",
            f"<td>{fmt_pct(row['new_retention'])}</td>",
            f"<td class='delta'>{fmt_delta(row['new_delta'])}</td>",
            f"<td>{fmt_int(row['old_reusable'])}</td>",
            f"<td class='coverage'>{fmt_pct(row['old_data_coverage_pct'])}</td>",
            f"<td class='delta'>{fmt_delta(row['old_reuse_delta'])}</td>",
            f"<td>{fmt_pct(row['old_matched_retention_pct'])}</td>",
            "</tr>",
        ]

    parts += [
        "</tbody></table></div>",
        "<div class='info'>",
        "<h2>Matching diagnostics</h2>",
        "<div class='info-grid'>",
    ]

    diagnostic_items = [
        ("Old duplicate raw pairs", metadata["old_duplicate_raw_pairs"]),
        ("Old rows missing root/id", metadata["old_missing_root_id"]),
        ("Old source IDs absent from NEW source_ids", metadata["old_root_not_in_source_ids"]),
        ("Old rows missing language", metadata["old_missing_language"]),
        ("Old languages absent from metadata", metadata["old_language_not_in_metadata"]),
        ("Old ambiguous language names", metadata["old_ambiguous_language"]),
        ("New rows missing root/nllb_code", metadata["new_missing_pair_key"]),
    ]

    for label, value in diagnostic_items:
        parts.append(
            f"<div><strong>{html.escape(label)}:</strong> "
            f"{int(value):,}</div>"
        )

    parts += [
        "</div></div>",
        "<div class='info'>",
        "<h2>Input files</h2>",
        f"<p><strong>OLD data:</strong> {html.escape(str(old_file))}</p>",
        f"<p><strong>NEW canonical roots:</strong> {html.escape(str(new_roots_file))}</p>",
        f"<p><strong>Language metadata:</strong> {html.escape(str(language_metadata_file))}</p>",
        "<p><strong>NEW quality files:</strong></p><ul>",
    ]

    for path in new_files:
        parts.append(f"<li>{html.escape(str(path))}</li>")

    parts += [
        "</ul>",
        "</div>",
        "</body></html>",
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        "\n".join(parts),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate BERT/COMET/chrF percentile analysis and OLD->NEW "
            "reuse coverage using source_ids + language metadata."
        )
    )

    parser.add_argument(
        "--new-dir",
        type=Path,
        required=True,
        help="Directory containing NEW translation-quality JSONL shards.",
    )
    parser.add_argument(
        "--new-pattern",
        required=True,
        help=(
            "Glob pattern for NEW quality JSONLs, e.g. "
            "'stage06_unsafe_roots_part_*_multilang_translation_quality.jsonl'."
        ),
    )
    parser.add_argument(
        "--old-file",
        type=Path,
        required=True,
        help="OLD multilingual JSONL file.",
    )
    parser.add_argument(
        "--new-roots-file",
        type=Path,
        required=True,
        help=(
            "NEW canonical roots JSONL containing root_id and source_ids, "
            "e.g. stage05_unique_roots.jsonl."
        ),
    )
    parser.add_argument(
        "--language-metadata-file",
        type=Path,
        required=True,
        help="Path to config/language_metadata.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output HTML report.",
    )
    parser.add_argument(
        "--percentile-step",
        type=int,
        default=5,
        help="Percentile increment; must be a positive divisor of 100.",
    )

    args = parser.parse_args()

    if not args.new_dir.exists():
        raise FileNotFoundError(
            f"NEW data directory not found:\n{args.new_dir}"
        )

    if not args.old_file.exists():
        raise FileNotFoundError(
            f"OLD data file not found:\n{args.old_file}"
        )

    if not args.new_roots_file.exists():
        raise FileNotFoundError(
            f"NEW roots file not found:\n{args.new_roots_file}"
        )

    if not args.language_metadata_file.exists():
        raise FileNotFoundError(
            "Language metadata file not found:\n"
            f"{args.language_metadata_file}"
        )

    if (
        args.percentile_step <= 0
        or 100 % args.percentile_step != 0
    ):
        raise ValueError(
            "--percentile-step must be a positive divisor of 100."
        )

    new_files = sorted(
        args.new_dir.glob(args.new_pattern)
    )

    if not new_files:
        raise FileNotFoundError(
            f"No NEW files matching {args.new_pattern!r} "
            f"in {args.new_dir}"
        )

    print("\n" + "=" * 80)
    print("TRANSLATION QUALITY + OLD-DATA REUSE REPORT")
    print("=" * 80)
    print(f"JSON backend: {JSON_BACKEND}")
    print(f"NEW quality files found: {len(new_files)}")
    for path in new_files:
        print(f"  - {path.name}")

    # -----------------------------------------------------------------------
    # 1. Language name -> NLLB code
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LOADING LANGUAGE METADATA")
    print("=" * 80)

    language_module = load_language_metadata_module(
        args.language_metadata_file
    )
    language_entries = discover_language_entries(
        language_module
    )
    language_lookup, ambiguous_plain = build_language_lookup(
        language_entries
    )

    print(
        f"Language metadata entries discovered: {len(language_entries):,}"
    )
    print(
        f"Resolvable language lookup keys:       {len(language_lookup):,}"
    )
    print(
        f"Ambiguous plain language names:        {len(ambiguous_plain):,}"
    )

    # -----------------------------------------------------------------------
    # 2. OLD source ID -> canonical NEW root_id
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BUILDING SOURCE-ID -> CANONICAL ROOT CROSSWALK")
    print("=" * 80)

    source_to_canonical, root_stats = (
        build_source_to_canonical_root_map(
            args.new_roots_file
        )
    )

    print(
        f"Canonical root rows:                 "
        f"{root_stats['total_root_rows']:,}"
    )
    print(
        f"Unique source IDs mapped:            "
        f"{root_stats['unique_source_ids']:,}"
    )
    print(
        f"Roots missing source_ids:            "
        f"{root_stats['roots_missing_source_ids']:,}"
    )

    # -----------------------------------------------------------------------
    # 3. Map OLD corpus into canonical pair space
    # -----------------------------------------------------------------------
    old_info = read_and_map_old_data(
        old_file=args.old_file,
        source_to_canonical=source_to_canonical,
        language_lookup=language_lookup,
        ambiguous_plain=ambiguous_plain,
    )

    # -----------------------------------------------------------------------
    # 4. Load NEW quality corpus
    # -----------------------------------------------------------------------
    new_data = read_new_quality_data(
        new_files
    )

    # -----------------------------------------------------------------------
    # 5. Percentile + reuse analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPUTING PERCENTILES + OLD-DATA REUSE")
    print("=" * 80)

    rows, metadata = build_percentile_rows(
        new_data=new_data,
        old_info=old_info,
        percentile_step=args.percentile_step,
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

    for row in rows:
        new_delta = (
            0
            if row["new_delta"] is None
            else row["new_delta"]
        )
        reuse_delta = (
            0
            if row["old_reuse_delta"] is None
            else row["old_reuse_delta"]
        )

        print(
            f"{row['percentile']:>5} "
            f"{row['bert_threshold']:>10.6f} "
            f"{row['comet_threshold']:>10.6f} "
            f"{row['chrf_threshold']:>10.4f} "
            f"{row['new_count']:>12,} "
            f"{new_delta:>+12,} "
            f"{row['old_reusable']:>12,} "
            f"{row['old_data_coverage_pct']:>9.3f}% "
            f"{reuse_delta:>+12,}"
        )

    # -----------------------------------------------------------------------
    # 6. HTML
    # -----------------------------------------------------------------------
    generate_html(
        rows=rows,
        metadata=metadata,
        new_files=new_files,
        old_file=args.old_file,
        new_roots_file=args.new_roots_file,
        language_metadata_file=args.language_metadata_file,
        output_file=args.output,
        percentile_step=args.percentile_step,
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"\nHTML report:\n{args.output}")
    print(f"\nOpen with:\nxdg-open '{args.output}'")


if __name__ == "__main__":
    main()