#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build the canonical unique-root AEGIS dataset.

Stages
------
00 - Load and merge train / validation / test
01 - Structural validity
02 - Remove REDACTED-only prompts
03 - Keep valid human prompt annotations
04 - Normalize prompts and detect safe/unsafe label conflicts
05 - Collapse label-consistent duplicate prompts

Final output
------------
stage05_unique_roots.jsonl

Important
---------
- Safe and unsafe rows remain together through Stage 05.
- Human safe/unsafe conflicts are excluded.
- Exact normalized duplicate prompts are collapsed.
- Categories from duplicate rows are unioned.
- Stable root_id values are generated from normalized prompts.
"""

import argparse
import hashlib
import json
import re
import unicodedata

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import load_dataset


DATASET_NAME = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"

AEGIS_FIELDS = [
    "id",
    "reconstruction_id_if_redacted",
    "prompt",
    "response",
    "prompt_label",
    "response_label",
    "violated_categories",
    "prompt_label_source",
    "response_label_source",
]


# =============================================================================
# Generic helpers
# =============================================================================

def is_missing(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, str):
        return not value.strip()

    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0

    return False


def normalize_prompt(text: str) -> str:
    """
    Conservative normalization for exact-root deduplication.

    - Unicode NFKC
    - strip surrounding whitespace
    - collapse repeated whitespace
    - preserve case
    """

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text.strip())

    return text


def root_id_from_prompt(normalized_prompt: str) -> str:
    digest = hashlib.sha1(
        normalized_prompt.encode("utf-8")
    ).hexdigest()[:16]

    return f"aegis_root_{digest}"


def is_exact_redaction(text: str) -> bool:
    if not isinstance(text, str):
        return False

    return text.strip().upper() in {
        "REDACTED",
        "[REDACTED]",
        "<REDACTED>",
    }


def contains_redaction(text: str) -> bool:
    if not isinstance(text, str):
        return False

    return "REDACTED" in text.upper()


def parse_categories(value: Any) -> List[str]:
    """
    Support categories stored as either a list or comma-separated string.
    """

    if value is None:
        return []

    if isinstance(value, str):
        categories = value.split(",")

    elif isinstance(value, (list, tuple, set)):
        categories = list(value)

    else:
        return []

    result = []

    for category in categories:

        if category is None:
            continue

        category = str(category).strip()

        if category:
            result.append(category)

    return sorted(set(result))


def write_jsonl(
    path: Path,
    rows: Iterable[Dict[str, Any]],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_json(
    path: Path,
    obj: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


# =============================================================================
# Stage 00
# =============================================================================

def stage00_load_all_splits(
    output_dir: Path,
):

    print("\n[Stage 00] Loading all AEGIS splits...")

    dataset = load_dataset(
        DATASET_NAME
    )

    rows = []
    split_counts = {}

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        if split_name not in dataset:
            print(
                f"WARNING: {split_name} split not found."
            )
            continue

        split = dataset[split_name]

        split_counts[
            split_name
        ] = len(split)

        for row in split:

            record = {
                field: row.get(field)
                for field in AEGIS_FIELDS
            }

            record[
                "aegis_split"
            ] = split_name

            rows.append(record)

    write_jsonl(
        output_dir / "stage00_raw.jsonl",
        rows,
    )

    stats = {
        "total_rows": len(rows),
        "split_counts": split_counts,
    }

    print(stats)

    return rows, stats


# =============================================================================
# Stage 01
# =============================================================================

def stage01_structural(
    rows,
    output_dir: Path,
):

    print("\n[Stage 01] Structural validation...")

    kept = []
    excluded = []

    reasons = Counter()

    for row in rows:

        row_reasons = []

        if is_missing(
            row.get("id")
        ):
            row_reasons.append(
                "missing_id"
            )

        if is_missing(
            row.get("prompt")
        ):
            row_reasons.append(
                "missing_prompt"
            )

        if is_missing(
            row.get("prompt_label")
        ):
            row_reasons.append(
                "missing_prompt_label"
            )

        if row_reasons:

            r = dict(row)

            r[
                "_exclusion_reasons"
            ] = row_reasons

            excluded.append(r)

            reasons.update(
                row_reasons
            )

            continue

        kept.append(row)

    write_jsonl(
        output_dir /
        "stage01_structural.jsonl",
        kept,
    )

    write_jsonl(
        output_dir /
        "stage01_structural_excluded.jsonl",
        excluded,
    )

    stats = {
        "input_rows":
            len(rows),

        "kept_rows":
            len(kept),

        "excluded_rows":
            len(excluded),

        "exclusion_reasons":
            dict(reasons),
    }

    print(stats)

    return kept, stats


# =============================================================================
# Stage 02
# =============================================================================

def stage02_redaction(
    rows,
    output_dir: Path,
):

    print("\n[Stage 02] Redaction analysis...")

    kept = []

    exact_redaction = []
    partial_redaction = []

    for row in rows:

        prompt = row["prompt"]

        if is_exact_redaction(
            prompt
        ):
            exact_redaction.append(
                row
            )
            continue

        r = dict(row)

        r[
            "contains_redaction"
        ] = contains_redaction(
            prompt
        )

        if r[
            "contains_redaction"
        ]:
            partial_redaction.append(
                r
            )

        kept.append(r)

    write_jsonl(
        output_dir /
        "stage02_nonredacted.jsonl",
        kept,
    )

    write_jsonl(
        output_dir /
        "stage02_exact_redaction_excluded.jsonl",
        exact_redaction,
    )

    write_jsonl(
        output_dir /
        "stage02_partial_redaction_flagged.jsonl",
        partial_redaction,
    )

    stats = {
        "input_rows":
            len(rows),

        "kept_rows":
            len(kept),

        "exact_redaction_removed":
            len(exact_redaction),

        "partial_redaction_flagged":
            len(partial_redaction),
    }

    print(stats)

    return kept, stats


# =============================================================================
# Stage 03
# =============================================================================

def stage03_human_labels(
    rows,
    output_dir: Path,
):

    print(
        "\n[Stage 03] Human prompt-label validation..."
    )

    source_distribution = Counter()

    for row in rows:

        source = str(
            row.get(
                "prompt_label_source"
            )
            or "<missing>"
        ).strip().lower()

        label = str(
            row.get(
                "prompt_label"
            )
            or "<missing>"
        ).strip().lower()

        source_distribution[
            f"{source}|{label}"
        ] += 1

    kept = []
    excluded = []

    reasons = Counter()

    for row in rows:

        source = str(
            row.get(
                "prompt_label_source"
            )
            or ""
        ).strip().lower()

        label = str(
            row.get(
                "prompt_label"
            )
            or ""
        ).strip().lower()

        row_reasons = []

        if label not in {
            "safe",
            "unsafe",
        }:
            row_reasons.append(
                "invalid_prompt_label"
            )

        if source != "human":
            row_reasons.append(
                "prompt_label_not_human"
            )

        if row_reasons:

            r = dict(row)

            r[
                "_exclusion_reasons"
            ] = row_reasons

            excluded.append(r)

            reasons.update(
                row_reasons
            )

            continue

        r = dict(row)

        r[
            "prompt_label"
        ] = label

        r[
            "prompt_label_source"
        ] = "human"

        kept.append(r)

    write_jsonl(
        output_dir /
        "stage03_human_labeled.jsonl",
        kept,
    )

    write_jsonl(
        output_dir /
        "stage03_annotation_excluded.jsonl",
        excluded,
    )

    stats = {
        "input_rows":
            len(rows),

        "kept_rows":
            len(kept),

        "excluded_rows":
            len(excluded),

        "source_label_distribution":
            dict(
                source_distribution
            ),

        "exclusion_reasons":
            dict(reasons),
    }

    print(stats)

    return kept, stats


# =============================================================================
# Stage 04
# =============================================================================

def stage04_conflicts(
    rows,
    output_dir: Path,
):

    print(
        "\n[Stage 04] Duplicate and label-conflict analysis..."
    )

    grouped = defaultdict(
        list
    )

    for row in rows:

        normalized = normalize_prompt(
            row["prompt"]
        )

        r = dict(row)

        r[
            "normalized_prompt"
        ] = normalized

        grouped[
            normalized
        ].append(r)

    duplicate_rows = []
    conflict_rows = []

    conflict_roots = {}

    clean_rows = []

    duplicate_size_distribution = Counter()

    duplicate_root_count = 0

    for normalized_prompt, group in grouped.items():

        duplicate_size_distribution[
            len(group)
        ] += 1

        if len(group) > 1:

            duplicate_root_count += 1

            duplicate_rows.extend(
                group
            )

        labels = {
            str(
                row[
                    "prompt_label"
                ]
            ).strip().lower()
            for row in group
        }

        # Safe/unsafe human annotation conflict
        if len(labels) > 1:

            conflict_roots[
                normalized_prompt
            ] = {
                "labels":
                    sorted(labels),

                "num_rows":
                    len(group),
            }

            for row in group:

                r = dict(row)

                r[
                    "_conflicting_labels"
                ] = sorted(labels)

                conflict_rows.append(
                    r
                )

            continue

        clean_rows.extend(
            group
        )

    write_jsonl(
        output_dir /
        "stage04_nonconflict_rows.jsonl",
        clean_rows,
    )

    write_jsonl(
        output_dir /
        "stage04_duplicate_rows.jsonl",
        duplicate_rows,
    )

    write_jsonl(
        output_dir /
        "stage04_label_conflict_rows.jsonl",
        conflict_rows,
    )

    write_json(
        output_dir /
        "stage04_label_conflict_roots.json",
        conflict_roots,
    )

    stats = {
        "input_rows":
            len(rows),

        "unique_normalized_prompts":
            len(grouped),

        "duplicate_root_prompts":
            duplicate_root_count,

        "label_conflict_root_prompts":
            len(
                conflict_roots
            ),

        "label_conflict_rows":
            len(
                conflict_rows
            ),

        "nonconflict_rows":
            len(
                clean_rows
            ),

        "duplicate_size_distribution": {
            str(k): v
            for k, v
            in sorted(
                duplicate_size_distribution.items()
            )
        },
    }

    print(stats)

    return clean_rows, stats


# =============================================================================
# Stage 05
# =============================================================================

def stage05_collapse_roots(
    rows,
    output_dir: Path,
):

    print(
        "\n[Stage 05] Collapsing duplicate roots..."
    )

    grouped = defaultdict(
        list
    )

    for row in rows:

        grouped[
            row[
                "normalized_prompt"
            ]
        ].append(row)

    roots = []

    for normalized_prompt, group in grouped.items():

        labels = {
            row[
                "prompt_label"
            ]
            for row in group
        }

        if len(labels) != 1:

            raise RuntimeError(
                "Unexpected label conflict for "
                f"{normalized_prompt}"
            )

        label = next(
            iter(labels)
        )

        # ---------------------------------------------------------------------
        # Union categories across duplicate source rows.
        # ---------------------------------------------------------------------

        category_union = set()

        for row in group:

            category_union.update(
                parse_categories(
                    row.get(
                        "violated_categories"
                    )
                )
            )

        # ---------------------------------------------------------------------
        # Preserve source provenance.
        # ---------------------------------------------------------------------

        source_ids = sorted({
            str(
                row["id"]
            )
            for row in group
        })

        source_splits = sorted({
            row[
                "aegis_split"
            ]
            for row in group
        })

        reconstruction_ids = sorted({
            str(
                row[
                    "reconstruction_id_if_redacted"
                ]
            )
            for row in group
            if not is_missing(
                row.get(
                    "reconstruction_id_if_redacted"
                )
            )
        })

        response_labels = sorted({
            str(
                row.get(
                    "response_label"
                )
            ).strip().lower()

            for row in group

            if not is_missing(
                row.get(
                    "response_label"
                )
            )
        })

        response_label_sources = sorted({
            str(
                row.get(
                    "response_label_source"
                )
            ).strip().lower()

            for row in group

            if not is_missing(
                row.get(
                    "response_label_source"
                )
            )
        })

        representative = group[0]

        root = {
            "root_id":
                root_id_from_prompt(
                    normalized_prompt
                ),

            "prompt":
                representative[
                    "prompt"
                ],

            "normalized_prompt":
                normalized_prompt,

            "prompt_label":
                label,

            "prompt_label_source":
                "human",

            "violated_categories":
                sorted(
                    category_union
                ),

            "num_source_rows":
                len(group),

            "source_ids":
                source_ids,

            "aegis_splits":
                source_splits,

            "reconstruction_ids":
                reconstruction_ids,

            "response_labels_observed":
                response_labels,

            "response_label_sources_observed":
                response_label_sources,

            "contains_redaction":
                any(
                    row.get(
                        "contains_redaction",
                        False,
                    )
                    for row in group
                ),
        }

        roots.append(
            root
        )

    # Stable output ordering
    roots = sorted(
        roots,
        key=lambda x: x[
            "root_id"
        ],
    )

    write_jsonl(
        output_dir /
        "stage05_unique_roots.jsonl",
        roots,
    )

    stats = {
        "input_rows":
            len(rows),

        "unique_root_prompts":
            len(roots),

        "rows_collapsed":
            len(rows)
            - len(roots),

        "safe_roots":
            sum(
                row[
                    "prompt_label"
                ] == "safe"
                for row in roots
            ),

        "unsafe_roots":
            sum(
                row[
                    "prompt_label"
                ] == "unsafe"
                for row in roots
            ),
    }

    print(stats)

    return roots, stats


# =============================================================================
# Main
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build canonical AEGIS unique roots "
            "through Stage 05."
        )
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directory for Stage 00-05 outputs."
        ),
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_stats = {}

    # Stage 00
    rows, stats = (
        stage00_load_all_splits(
            output_dir
        )
    )

    all_stats[
        "stage00"
    ] = stats

    # Stage 01
    rows, stats = (
        stage01_structural(
            rows,
            output_dir,
        )
    )

    all_stats[
        "stage01"
    ] = stats

    # Stage 02
    rows, stats = (
        stage02_redaction(
            rows,
            output_dir,
        )
    )

    all_stats[
        "stage02"
    ] = stats

    # Stage 03
    rows, stats = (
        stage03_human_labels(
            rows,
            output_dir,
        )
    )

    all_stats[
        "stage03"
    ] = stats

    # Stage 04
    rows, stats = (
        stage04_conflicts(
            rows,
            output_dir,
        )
    )

    all_stats[
        "stage04"
    ] = stats

    # Stage 05
    roots, stats = (
        stage05_collapse_roots(
            rows,
            output_dir,
        )
    )

    all_stats[
        "stage05"
    ] = stats

    # -------------------------------------------------------------------------
    # Save Stage 00-05 audit.
    # -------------------------------------------------------------------------

    audit_file = (
        output_dir
        / "aegis_root_construction_audit.json"
    )

    write_json(
        audit_file,
        all_stats,
    )

    print()
    print("=" * 76)
    print(
        "AEGIS ROOT CONSTRUCTION COMPLETE"
    )
    print("=" * 76)

    print(
        f"Final unique roots: "
        f"{len(roots):,}"
    )

    print(
        f"Stage 05 dataset: "
        f"{output_dir / 'stage05_unique_roots.jsonl'}"
    )






    print(
        f"Audit: "
        f"{audit_file}"
    )


if __name__ == "__main__":
    main()