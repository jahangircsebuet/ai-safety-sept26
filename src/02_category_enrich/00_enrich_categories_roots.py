#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add:
    core_categories
    finegrained_categories

to each AEGIS root object using violated_categories.

Also creates an audit log containing:
    1. root_ids with neither core nor fine-grained categories
    2. unknown category names and the root_ids containing them

Input:
    JSONL or JSON array

Output:
    JSONL

Log:
    Automatically created beside the output file:
        <output_filename>.log
"""

import argparse
import json

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


# =============================================================================
# CATEGORY DEFINITIONS
# =============================================================================

CORE_CATEGORIES = {
    "Hate/Identity Hate",
    "Suicide and Self Harm",
    "Guns and Illegal Weapons",
    "PII/Privacy",
    "Criminal Planning/Confessions",
    "Controlled/Regulated Substances",
    "Sexual",
    "Violence",
    "Threat",
    "Sexual (minor)",
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
    "High Risk Gov Decision Making",
    "Malware",
    "Manipulation",
}


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
    # Try normal JSON first
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
                row = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line "
                    f"{line_number}: {exc}"
                )

            if not isinstance(row, dict):
                raise ValueError(
                    f"Line {line_number} is not "
                    f"a JSON object."
                )

            rows.append(row)

    return rows


# =============================================================================
# CATEGORY ENRICHMENT
# =============================================================================

def add_category_groups(
    row: Dict[str, Any]
) -> Dict[str, Any]:

    violated_categories = row.get(
        "violated_categories",
        []
    )

    if violated_categories is None:
        violated_categories = []

    if not isinstance(
        violated_categories,
        list
    ):
        raise ValueError(
            f"violated_categories must be a list "
            f"for root_id={row.get('root_id')}"
        )

    # -------------------------------------------------------------------------
    # Preserve original category ordering
    # -------------------------------------------------------------------------

    core_categories = [
        category
        for category in violated_categories
        if category in CORE_CATEGORIES
    ]

    finegrained_categories = [
        category
        for category in violated_categories
        if category in FINE_GRAINED_CATEGORIES
    ]

    row["core_categories"] = (
        core_categories
    )

    row["finegrained_categories"] = (
        finegrained_categories
    )

    return row


# =============================================================================
# MAIN PROCESS
# =============================================================================

def process_file(
    input_file: Path,
    output_file: Path
) -> None:

    print("=" * 80)
    print("ADD CORE / FINE-GRAINED CATEGORIES")
    print("=" * 80)

    rows = load_json_or_jsonl(
        input_file
    )

    print(
        f"Input rows: {len(rows):,}"
    )

    output_rows = []

    # -------------------------------------------------------------------------
    # Audit structures
    # -------------------------------------------------------------------------

    # category -> set(root_ids)
    unknown_category_root_ids = defaultdict(set)

    # Store details for rows belonging to neither taxonomy.
    neither_rows = []

    core_count = 0
    fine_count = 0
    mixed_count = 0
    no_group_count = 0

    # =========================================================================
    # PROCESS ROWS
    # =========================================================================

    for row in rows:

        root_id = str(
            row.get(
                "root_id",
                "MISSING_ROOT_ID"
            )
        )

        violated = row.get(
            "violated_categories",
            []
        )

        if violated is None:
            violated = []

        # ---------------------------------------------------------------------
        # Record unknown category names + corresponding root IDs
        # ---------------------------------------------------------------------

        for category in violated:

            if (
                category not in CORE_CATEGORIES
                and
                category not in FINE_GRAINED_CATEGORIES
            ):

                unknown_category_root_ids[
                    category
                ].add(
                    root_id
                )

        # ---------------------------------------------------------------------
        # Add new category-group fields
        # ---------------------------------------------------------------------

        row = add_category_groups(
            row
        )

        has_core = bool(
            row["core_categories"]
        )

        has_fine = bool(
            row["finegrained_categories"]
        )

        if has_core:
            core_count += 1

        if has_fine:
            fine_count += 1

        if has_core and has_fine:
            mixed_count += 1

        # ---------------------------------------------------------------------
        # Neither core nor fine-grained
        # ---------------------------------------------------------------------

        if not has_core and not has_fine:

            no_group_count += 1

            neither_rows.append({
                "root_id":
                    root_id,

                "prompt":
                    row.get(
                        "prompt"
                    ),

                "violated_categories":
                    violated,
            })

        output_rows.append(
            row
        )

    # =========================================================================
    # WRITE OUTPUT JSONL
    # =========================================================================

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        for row in output_rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
            )

            f.write("\n")

    # =========================================================================
    # LOG FILE
    # =========================================================================

    log_file = output_file.with_suffix(
        ".log"
    )

    with log_file.open(
        "w",
        encoding="utf-8"
    ) as log:

        # ---------------------------------------------------------------------
        # General summary
        # ---------------------------------------------------------------------

        log.write(
            "=" * 80 + "\n"
        )

        log.write(
            "CATEGORY GROUPING AUDIT\n"
        )

        log.write(
            "=" * 80 + "\n\n"
        )

        log.write(
            f"Input rows: "
            f"{len(rows):,}\n"
        )

        log.write(
            f"Rows with core category: "
            f"{core_count:,}\n"
        )

        log.write(
            f"Rows with fine-grained category: "
            f"{fine_count:,}\n"
        )

        log.write(
            f"Rows containing both: "
            f"{mixed_count:,}\n"
        )

        log.write(
            f"Rows with neither: "
            f"{no_group_count:,}\n"
        )

        log.write(
            f"Unknown category names: "
            f"{len(unknown_category_root_ids):,}\n"
        )

        # =====================================================================
        # NEITHER CATEGORY
        # =====================================================================

        log.write("\n")
        log.write(
            "=" * 80 + "\n"
        )

        log.write(
            "ROOT IDS WITH NEITHER CORE NOR FINE-GRAINED CATEGORY\n"
        )

        log.write(
            "=" * 80 + "\n"
        )

        log.write(
            f"Count: {len(neither_rows):,}\n\n"
        )

        for item in neither_rows:

            log.write(
                f"root_id: "
                f"{item['root_id']}\n"
            )

            log.write(
                f"violated_categories: "
                f"{json.dumps(item['violated_categories'], ensure_ascii=False)}\n"
            )

            log.write(
                f"prompt: "
                f"{item['prompt']}\n"
            )

            log.write(
                "-" * 80 + "\n"
            )

        # =====================================================================
        # UNKNOWN CATEGORIES
        # =====================================================================

        log.write("\n")
        log.write(
            "=" * 80 + "\n"
        )

        log.write(
            "UNKNOWN CATEGORY NAMES AND ROOT IDS\n"
        )

        log.write(
            "=" * 80 + "\n"
        )

        log.write(
            f"Unknown categories: "
            f"{len(unknown_category_root_ids):,}\n\n"
        )

        for category in sorted(
            unknown_category_root_ids
        ):

            root_ids = sorted(
                unknown_category_root_ids[
                    category
                ]
            )

            log.write(
                f"CATEGORY: {category}\n"
            )

            log.write(
                f"ROOT COUNT: "
                f"{len(root_ids):,}\n"
            )

            for root_id in root_ids:

                log.write(
                    f"  {root_id}\n"
                )

            log.write(
                "-" * 80 + "\n"
            )

    # =========================================================================
    # CONSOLE SUMMARY
    # =========================================================================

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"Rows with core category:         "
        f"{core_count:,}"
    )

    print(
        f"Rows with fine-grained category: "
        f"{fine_count:,}"
    )

    print(
        f"Rows containing both:            "
        f"{mixed_count:,}"
    )

    print(
        f"Rows with neither:               "
        f"{no_group_count:,}"
    )

    print(
        f"Unknown category names:          "
        f"{len(unknown_category_root_ids):,}"
    )

    # -------------------------------------------------------------------------
    # Print unknown category summary to console as well
    # -------------------------------------------------------------------------

    if unknown_category_root_ids:

        print()
        print("Unknown categories:")

        for category in sorted(
            unknown_category_root_ids
        ):

            print(
                f"  - {category}: "
                f"{len(unknown_category_root_ids[category]):,} roots"
            )

    print()
    print(
        f"Saved dataset: {output_file}"
    )

    print(
        f"Saved audit log: {log_file}"
    )


# =============================================================================
# CLI
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Add core_categories and "
            "finegrained_categories to AEGIS roots."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON/JSONL file."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file."
    )

    args = parser.parse_args()

    process_file(
        input_file=Path(
            args.input
        ),
        output_file=Path(
            args.output
        )
    )


if __name__ == "__main__":
    main()