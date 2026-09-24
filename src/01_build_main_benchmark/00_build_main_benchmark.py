#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build the final AEGIS benchmark from Stage-05 unique canonical roots.

Input
-----
stage05_unique_roots.jsonl

Outputs
-------
1. stage06_main_benchmark_roots.jsonl
2. stage06_safe_roots.jsonl
3. stage06_unsafe_roots.jsonl
4. stage06_filtered_out_roots.jsonl
5. stage06_benchmark_filter_audit.json


Final benchmark rules
---------------------

SAFE_MAIN
    prompt_label == "safe"
    AND
    violated_categories == []

UNSAFE_MAIN
    prompt_label == "unsafe"
    AND
    at least one recognized core/fine-grained risk category
    AND
    does not contain "Needs Caution"
    AND
    does not contain "Other"
    AND
    does not contain any unmapped category

Everything else is retained in the filtered-out audit file.
"""

import argparse
import json

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


# =============================================================================
# CANONICAL TAXONOMY
# =============================================================================

CORE_CATEGORIES = {
    "Hate/Identity Hate",
    "Suicide and Self Harm",
    "Guns/Illegal Weapons",
    "PII/Privacy",
    "Criminal Planning/Confessions",
    "Controlled/Regulated Substances",
    "Sexual",
    "Violence",
    "Threat",
    "Sexual Minor",
    "Harassment",
    "Profanity",
}


FINE_GRAINED_CATEGORIES = {
    "Illegal Activity",
    "Immoral/Unethical",
    "Unauthorized Advice",
    "Political/Misinformation/Conspiracy",
    "Fraud/Deception",
    "Copyright/Trademark/Plagiarism",
    "High Risk Gov. Decision Making",
    "Malware",
    "Manipulation",
}


SPECIAL_EXCLUDED_CATEGORIES = {
    "Needs Caution",
    "Other",
}


# =============================================================================
# LEGACY / SPELLING ALIASES
# =============================================================================
#
# This is important because your old/new AEGIS data may contain slightly
# different spellings.
#
# Everything is normalized to the canonical names above.
#

CATEGORY_ALIASES = {

    # Controlled substances
    "Controlled/Regulated substances":
        "Controlled/Regulated Substances",

    # Weapons
    "Guns and Illegal Weapons":
        "Guns/Illegal Weapons",

    # Government decision making
    "High Risk Gov Decision Making":
        "High Risk Gov. Decision Making",

    # Sexual minor
    "Sexual (minor)":
        "Sexual Minor",
}


# =============================================================================
# HELPERS
# =============================================================================

def normalize_category(
    category: Any
) -> str:
    """
    Normalize a category name while preserving the canonical taxonomy.
    """

    category = str(category).strip()

    return CATEGORY_ALIASES.get(
        category,
        category
    )


def deduplicate_preserve_order(
    values: List[str]
) -> List[str]:
    """Deduplicate a list without changing its original order."""

    seen = set()
    result = []

    for value in values:

        if value not in seen:

            seen.add(value)
            result.append(value)

    return result


# =============================================================================
# INPUT
# =============================================================================

def load_json_or_jsonl(
    path: Path
) -> List[Dict[str, Any]]:

    if not path.exists():

        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    # -------------------------------------------------------------------------
    # First attempt normal JSON
    # -------------------------------------------------------------------------

    try:

        with path.open(
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            return [data]

    except json.JSONDecodeError:
        pass

    # -------------------------------------------------------------------------
    # Fall back to JSONL
    # -------------------------------------------------------------------------

    rows = []

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:

                row = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON at line "
                    f"{line_number}: {exc}"
                )

            if not isinstance(
                row,
                dict
            ):

                raise ValueError(
                    f"Line {line_number} "
                    "is not a JSON object."
                )

            rows.append(
                row
            )

    return rows


# =============================================================================
# OUTPUT
# =============================================================================

def write_jsonl(
    path: Path,
    rows: List[Dict[str, Any]]
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        encoding="utf-8"
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
            )

            f.write("\n")


# =============================================================================
# TAXONOMY ENRICHMENT
# =============================================================================

def enrich_categories(
    row: Dict[str, Any]
) -> Dict[str, Any]:

    violated = row.get(
        "violated_categories",
        []
    )

    if violated is None:
        violated = []

    if not isinstance(
        violated,
        list
    ):

        raise ValueError(
            f"violated_categories must be a list "
            f"for root_id={row.get('root_id')}"
        )

    # -------------------------------------------------------------------------
    # Normalize known aliases
    # -------------------------------------------------------------------------

    normalized = [
        normalize_category(category)
        for category in violated
    ]

    normalized = (
        deduplicate_preserve_order(
            normalized
        )
    )

    # -------------------------------------------------------------------------
    # Split taxonomy
    # -------------------------------------------------------------------------

    core_categories = [
        category
        for category in normalized
        if category in CORE_CATEGORIES
    ]

    finegrained_categories = [
        category
        for category in normalized
        if category in FINE_GRAINED_CATEGORIES
    ]

    needs_caution = (
        "Needs Caution"
        in normalized
    )

    has_other_category = (
        "Other"
        in normalized
    )

    known = (
        CORE_CATEGORIES
        | FINE_GRAINED_CATEGORIES
        | SPECIAL_EXCLUDED_CATEGORIES
    )

    unmapped_categories = [
        category
        for category in normalized
        if category not in known
    ]

    risk_categories = (
        core_categories
        + finegrained_categories
    )

    # -------------------------------------------------------------------------
    # Add enriched fields
    # -------------------------------------------------------------------------

    row["violated_categories"] = normalized

    row["core_categories"] = (
        core_categories
    )

    row["finegrained_categories"] = (
        finegrained_categories
    )

    row["risk_categories"] = (
        risk_categories
    )

    row["needs_caution"] = (
        needs_caution
    )

    row["has_other_category"] = (
        has_other_category
    )

    row["unmapped_categories"] = (
        unmapped_categories
    )

    return row


# =============================================================================
# BENCHMARK CLASSIFICATION
# =============================================================================

def classify_benchmark_row(
    row: Dict[str, Any]
) -> Dict[str, Any]:

    row = enrich_categories(
        row
    )

    prompt_label = str(
        row.get(
            "prompt_label",
            ""
        )
    ).strip().lower()

    violated = row[
        "violated_categories"
    ]

    risk_categories = row[
        "risk_categories"
    ]

    needs_caution = row[
        "needs_caution"
    ]

    has_other = row[
        "has_other_category"
    ]

    unmapped = row[
        "unmapped_categories"
    ]

    filter_reasons = []

    benchmark_partition = None

    is_main_benchmark = False

    # =========================================================================
    # SAFE
    # =========================================================================

    if prompt_label == "safe":

        # Clean safe control
        if len(violated) == 0:

            benchmark_partition = (
                "safe_main"
            )

            is_main_benchmark = True

        else:

            filter_reasons.append(
                "safe_with_category_annotation"
            )

            if needs_caution:

                filter_reasons.append(
                    "contains_needs_caution"
                )

            if has_other:

                filter_reasons.append(
                    "contains_other"
                )

            if unmapped:

                filter_reasons.append(
                    "contains_unmapped_category"
                )

    # =========================================================================
    # UNSAFE
    # =========================================================================

    elif prompt_label == "unsafe":

        # ---------------------------------------------------------------------
        # Explicit exclusions first
        # ---------------------------------------------------------------------

        if needs_caution:

            filter_reasons.append(
                "contains_needs_caution"
            )

        if has_other:

            filter_reasons.append(
                "contains_other"
            )

        if unmapped:

            filter_reasons.append(
                "contains_unmapped_category"
            )

        if len(risk_categories) == 0:

            filter_reasons.append(
                "unsafe_without_known_risk_category"
            )

        # ---------------------------------------------------------------------
        # Keep only completely unambiguous unsafe roots
        # ---------------------------------------------------------------------

        if len(filter_reasons) == 0:

            benchmark_partition = (
                "unsafe_main"
            )

            is_main_benchmark = True

    # =========================================================================
    # INVALID / UNEXPECTED LABEL
    # =========================================================================

    else:

        filter_reasons.append(
            "invalid_prompt_label"
        )

    # -------------------------------------------------------------------------
    # Primary exclusion reason
    # -------------------------------------------------------------------------

    primary_exclusion_reason = None

    if not is_main_benchmark:

        # Priority ordering makes primary reason deterministic.
        priority = [
            "contains_needs_caution",
            "contains_other",
            "contains_unmapped_category",
            "safe_with_category_annotation",
            "unsafe_without_known_risk_category",
            "invalid_prompt_label",
        ]

        for reason in priority:

            if reason in filter_reasons:

                primary_exclusion_reason = (
                    reason
                )

                break

    # -------------------------------------------------------------------------
    # Add fields
    # -------------------------------------------------------------------------

    row["benchmark_partition"] = (
        benchmark_partition
    )

    row["is_main_benchmark"] = (
        is_main_benchmark
    )

    row["filter_reasons"] = (
        filter_reasons
    )

    row["primary_exclusion_reason"] = (
        primary_exclusion_reason
    )

    return row


# =============================================================================
# MAIN
# =============================================================================

def build_benchmark(
    input_file: Path,
    output_dir: Path
) -> None:

    print("=" * 80)
    print("STAGE 06 — BUILD FINAL AEGIS BENCHMARK")
    print("=" * 80)

    print(
        f"Input: {input_file}"
    )

    print(
        f"Output directory: {output_dir}"
    )

    # =========================================================================
    # Load Stage 05
    # =========================================================================

    rows = load_json_or_jsonl(
        input_file
    )

    print()
    print(
        f"Stage-05 roots: {len(rows):,}"
    )

    # =========================================================================
    # Process
    # =========================================================================

    main_rows = []

    safe_rows = []

    unsafe_rows = []

    filtered_rows = []

    input_label_counts = Counter()

    input_category_counts = Counter()

    benchmark_category_counts = Counter()

    exclusion_reason_counts = Counter()

    all_filter_reason_counts = Counter()

    risk_count_distribution = Counter()

    # -------------------------------------------------------------------------
    # Integrity
    # -------------------------------------------------------------------------

    duplicate_root_ids = 0

    seen_root_ids = set()

    # =========================================================================
    # Loop
    # =========================================================================

    for row in rows:

        root_id = row.get(
            "root_id"
        )

        if root_id in seen_root_ids:

            duplicate_root_ids += 1

        seen_root_ids.add(
            root_id
        )

        # ---------------------------------------------------------------------
        # Input statistics
        # ---------------------------------------------------------------------

        prompt_label = str(
            row.get(
                "prompt_label",
                ""
            )
        ).strip().lower()

        input_label_counts[
            prompt_label
        ] += 1

        original_violated = row.get(
            "violated_categories",
            []
        ) or []

        for category in original_violated:

            input_category_counts[
                normalize_category(
                    category
                )
            ] += 1

        # ---------------------------------------------------------------------
        # Benchmark classification
        # ---------------------------------------------------------------------

        row = classify_benchmark_row(
            row
        )

        # ---------------------------------------------------------------------
        # MAIN BENCHMARK
        # ---------------------------------------------------------------------

        if row[
            "is_main_benchmark"
        ]:

            main_rows.append(
                row
            )

            if (
                row[
                    "benchmark_partition"
                ]
                == "safe_main"
            ):

                safe_rows.append(
                    row
                )

            elif (
                row[
                    "benchmark_partition"
                ]
                == "unsafe_main"
            ):

                unsafe_rows.append(
                    row
                )

                # Category counts only make sense for unsafe rows.
                for category in row[
                    "risk_categories"
                ]:

                    benchmark_category_counts[
                        category
                    ] += 1

            risk_count_distribution[
                len(
                    row[
                        "risk_categories"
                    ]
                )
            ] += 1

        # ---------------------------------------------------------------------
        # FILTERED OUT
        # ---------------------------------------------------------------------

        else:

            filtered_rows.append(
                row
            )

            primary_reason = row[
                "primary_exclusion_reason"
            ]

            exclusion_reason_counts[
                primary_reason
            ] += 1

            for reason in row[
                "filter_reasons"
            ]:

                all_filter_reason_counts[
                    reason
                ] += 1

    # =========================================================================
    # OUTPUT PATHS
    # =========================================================================

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    main_file = (
        output_dir
        / "stage06_main_benchmark_roots.jsonl"
    )

    safe_file = (
        output_dir
        / "stage06_safe_roots.jsonl"
    )

    unsafe_file = (
        output_dir
        / "stage06_unsafe_roots.jsonl"
    )

    filtered_file = (
        output_dir
        / "stage06_filtered_out_roots.jsonl"
    )

    audit_file = (
        output_dir
        / "stage06_benchmark_filter_audit.json"
    )

    # =========================================================================
    # WRITE DATASETS
    # =========================================================================

    write_jsonl(
        main_file,
        main_rows
    )

    write_jsonl(
        safe_file,
        safe_rows
    )

    write_jsonl(
        unsafe_file,
        unsafe_rows
    )

    write_jsonl(
        filtered_file,
        filtered_rows
    )

    # =========================================================================
    # INTEGRITY CHECKS
    # =========================================================================

    main_root_ids = {
        row["root_id"]
        for row in main_rows
    }

    filtered_root_ids = {
        row["root_id"]
        for row in filtered_rows
    }

    safe_root_ids = {
        row["root_id"]
        for row in safe_rows
    }

    unsafe_root_ids = {
        row["root_id"]
        for row in unsafe_rows
    }

    integrity = {

        "input_equals_main_plus_filtered":
            len(rows)
            ==
            (
                len(main_rows)
                +
                len(filtered_rows)
            ),

        "main_filtered_overlap":
            len(
                main_root_ids
                & filtered_root_ids
            ),

        "safe_unsafe_overlap":
            len(
                safe_root_ids
                & unsafe_root_ids
            ),

        "main_equals_safe_plus_unsafe":
            len(main_rows)
            ==
            (
                len(safe_rows)
                +
                len(unsafe_rows)
            ),

        "duplicate_root_ids_input":
            duplicate_root_ids,
    }

    # =========================================================================
    # AUDIT
    # =========================================================================

    audit = {

        "input": {
            "total_roots":
                len(rows),

            "prompt_label_counts":
                dict(
                    sorted(
                        input_label_counts.items()
                    )
                ),

            "category_counts":
                dict(
                    sorted(
                        input_category_counts.items()
                    )
                ),
        },

        "main_benchmark": {
            "total_roots":
                len(main_rows),

            "safe_main":
                len(safe_rows),

            "unsafe_main":
                len(unsafe_rows),

            "retention_rate":
                (
                    len(main_rows)
                    / len(rows)
                    if rows
                    else 0.0
                ),

            "unsafe_category_counts":
                dict(
                    sorted(
                        benchmark_category_counts.items()
                    )
                ),

            "risk_count_distribution":
                {
                    str(k): v
                    for k, v in sorted(
                        risk_count_distribution.items()
                    )
                },
        },

        "filtered_out": {
            "total_roots":
                len(filtered_rows),

            "exclusion_rate":
                (
                    len(filtered_rows)
                    / len(rows)
                    if rows
                    else 0.0
                ),

            "primary_exclusion_reasons":
                dict(
                    sorted(
                        exclusion_reason_counts.items()
                    )
                ),

            "all_filter_reasons":
                dict(
                    sorted(
                        all_filter_reason_counts.items()
                    )
                ),
        },

        "taxonomy": {

            "core_categories":
                sorted(
                    CORE_CATEGORIES
                ),

            "finegrained_categories":
                sorted(
                    FINE_GRAINED_CATEGORIES
                ),

            "special_excluded_categories":
                sorted(
                    SPECIAL_EXCLUDED_CATEGORIES
                ),

            "category_aliases":
                CATEGORY_ALIASES,
        },

        "integrity":
            integrity,
    }

    with audit_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            audit,
            f,
            ensure_ascii=False,
            indent=2
        )

    # =========================================================================
    # CONSOLE SUMMARY
    # =========================================================================

    print()
    print("=" * 80)
    print("STAGE 06 COMPLETE")
    print("=" * 80)

    print(
        f"Input Stage-05 roots     : "
        f"{len(rows):,}"
    )

    print()
    print(
        f"Final benchmark roots    : "
        f"{len(main_rows):,}"
    )

    print(
        f"  Safe roots             : "
        f"{len(safe_rows):,}"
    )

    print(
        f"  Unsafe roots           : "
        f"{len(unsafe_rows):,}"
    )

    print()
    print(
        f"Filtered-out roots       : "
        f"{len(filtered_rows):,}"
    )

    if rows:

        print(
            f"Retention rate           : "
            f"{100 * len(main_rows) / len(rows):.2f}%"
        )

        print(
            f"Exclusion rate           : "
            f"{100 * len(filtered_rows) / len(rows):.2f}%"
        )

    print()
    print("Primary exclusion reasons:")

    for reason, count in sorted(
        exclusion_reason_counts.items(),
        key=lambda x: (-x[1], str(x[0]))
    ):

        print(
            f"  {reason:40s} "
            f"{count:,}"
        )

    print()
    print("Integrity checks:")

    for key, value in integrity.items():

        print(
            f"  {key}: {value}"
        )

    print()
    print("Files written:")

    print(
        f"  {main_file}"
    )

    print(
        f"  {safe_file}"
    )

    print(
        f"  {unsafe_file}"
    )

    print(
        f"  {filtered_file}"
    )

    print(
        f"  {audit_file}"
    )


# =============================================================================
# CLI
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build the final safe/unsafe AEGIS benchmark "
            "from Stage-05 unique roots."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input Stage-05 unique-roots "
            "JSON/JSONL file."
        )
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directory where Stage-06 outputs "
            "will be written."
        )
    )

    args = parser.parse_args()

    build_benchmark(
        input_file=Path(
            args.input
        ),
        output_dir=Path(
            args.output_dir
        )
    )


if __name__ == "__main__":
    main()