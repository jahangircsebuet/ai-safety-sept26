#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split Stage-06 unsafe AEGIS roots into sequential JSONL files.

Example:
    11,788 rows with chunk_size=2000

produces:

    stage06_unsafe_roots_part_01.jsonl   2000
    stage06_unsafe_roots_part_02.jsonl   2000
    stage06_unsafe_roots_part_03.jsonl   2000
    stage06_unsafe_roots_part_04.jsonl   2000
    stage06_unsafe_roots_part_05.jsonl   2000
    stage06_unsafe_roots_part_06.jsonl   1788

The original object structure is preserved exactly.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


# =============================================================================
# LOAD JSON / JSONL
# =============================================================================

def load_json_or_jsonl(
    path: Path
) -> List[Dict[str, Any]]:

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    # -------------------------------------------------------------------------
    # Try standard JSON first
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
                    f"Invalid JSON on line "
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
# WRITE JSONL
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
# SPLIT
# =============================================================================

def split_file(
    input_file: Path,
    output_dir: Path,
    chunk_size: int
) -> None:

    print("=" * 80)
    print("SPLIT STAGE-06 UNSAFE ROOTS")
    print("=" * 80)

    print(
        f"Input      : {input_file}"
    )

    print(
        f"Output dir : {output_dir}"
    )

    print(
        f"Chunk size : {chunk_size:,}"
    )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    rows = load_json_or_jsonl(
        input_file
    )

    total_rows = len(
        rows
    )

    num_parts = math.ceil(
        total_rows
        / chunk_size
    )

    print()
    print(
        f"Total objects: {total_rows:,}"
    )

    print(
        f"Number of parts: {num_parts:,}"
    )

    # -------------------------------------------------------------------------
    # Output directory
    # -------------------------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------------------------------
    # Use original filename stem
    # -------------------------------------------------------------------------

    base_name = input_file.stem

    written_total = 0

    output_files = []

    # =========================================================================
    # SPLIT
    # =========================================================================

    for part_index in range(
        num_parts
    ):

        start = (
            part_index
            * chunk_size
        )

        end = min(
            start + chunk_size,
            total_rows
        )

        part_rows = rows[
            start:end
        ]

        part_number = (
            part_index + 1
        )

        output_file = (
            output_dir
            / (
                f"{base_name}_part_"
                f"{part_number:02d}.jsonl"
            )
        )

        write_jsonl(
            output_file,
            part_rows
        )

        written_total += len(
            part_rows
        )

        output_files.append(
            (
                output_file,
                len(part_rows)
            )
        )

        print(
            f"Part {part_number:02d}: "
            f"{len(part_rows):,} rows "
            f"[{start:,}:{end:,}]"
        )

        print(
            f"    {output_file}"
        )

    # =========================================================================
    # INTEGRITY
    # =========================================================================

    if written_total != total_rows:

        raise RuntimeError(
            "Split integrity check failed: "
            f"input={total_rows:,}, "
            f"written={written_total:,}"
        )

    # Check root IDs did not disappear or duplicate during splitting.
    input_root_ids = [
        row.get("root_id")
        for row in rows
    ]

    unique_root_ids = set(
        input_root_ids
    )

    duplicate_root_ids = (
        total_rows
        - len(unique_root_ids)
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print()
    print("=" * 80)
    print("SPLIT COMPLETE")
    print("=" * 80)

    print(
        f"Input objects        : "
        f"{total_rows:,}"
    )

    print(
        f"Objects written      : "
        f"{written_total:,}"
    )

    print(
        f"Files created        : "
        f"{len(output_files):,}"
    )

    print(
        f"Unique root_ids      : "
        f"{len(unique_root_ids):,}"
    )

    print(
        f"Duplicate root_ids   : "
        f"{duplicate_root_ids:,}"
    )

    print()

    for path, count in output_files:

        print(
            f"{path.name}: {count:,}"
        )


# =============================================================================
# CLI
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Split unsafe Stage-06 AEGIS roots "
            "into JSONL chunks."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON/JSONL file."
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for split files."
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help=(
            "Number of objects per file. "
            "Default: 2000"
        )
    )

    args = parser.parse_args()

    if args.chunk_size <= 0:

        raise ValueError(
            "--chunk-size must be > 0"
        )

    split_file(
        input_file=Path(
            args.input
        ),
        output_dir=Path(
            args.output_dir
        ),
        chunk_size=args.chunk_size
    )


if __name__ == "__main__":
    main()