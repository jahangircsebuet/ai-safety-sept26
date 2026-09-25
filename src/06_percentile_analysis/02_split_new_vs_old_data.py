#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path


def split_by_sample_origin(
    input_file: Path,
    new_output: Path,
    old_output: Path,
    new_origin: str = "new_stratified",
) -> None:

    # Create output directories if necessary
    new_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    old_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = 0
    new_count = 0
    old_count = 0
    missing_origin = 0
    invalid_json = 0

    origin_counts = Counter()

    with (
        input_file.open(
            "r",
            encoding="utf-8",
        ) as fin,
        new_output.open(
            "w",
            encoding="utf-8",
        ) as fnew,
        old_output.open(
            "w",
            encoding="utf-8",
        ) as fold,
    ):

        for line_num, line in enumerate(
            fin,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            total += 1

            # ---------------------------------------------
            # Parse JSON
            # ---------------------------------------------

            try:
                obj = json.loads(line)

            except json.JSONDecodeError as e:

                invalid_json += 1

                print(
                    f"[WARN] Invalid JSON at "
                    f"line {line_num}: {e}"
                )

                continue

            # ---------------------------------------------
            # Get sample_origin
            # ---------------------------------------------

            sample_origin = obj.get(
                "sample_origin"
            )

            if sample_origin is None:

                missing_origin += 1

                print(
                    f"[WARN] Missing sample_origin "
                    f"at line {line_num}"
                )

                continue

            origin_counts[
                str(sample_origin)
            ] += 1

            # ---------------------------------------------
            # NEW DATA
            # ---------------------------------------------

            if sample_origin == new_origin:

                fnew.write(
                    json.dumps(
                        obj,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                new_count += 1

            # ---------------------------------------------
            # OLD DATA
            # ---------------------------------------------

            else:

                fold.write(
                    json.dumps(
                        obj,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                old_count += 1


    # =====================================================
    # SUMMARY
    # =====================================================

    print("\n========================================")
    print("SPLIT COMPLETE")
    print("========================================")

    print(
        f"Input rows          : {total:,}"
    )

    print(
        f"New rows            : {new_count:,}"
    )

    print(
        f"Old rows            : {old_count:,}"
    )

    print(
        f"Missing origin      : {missing_origin:,}"
    )

    print(
        f"Invalid JSON        : {invalid_json:,}"
    )

    print("\nObserved sample_origin values:")

    for origin, count in sorted(
        origin_counts.items()
    ):

        print(
            f"  {origin}: {count:,}"
        )

    print("\nOutput files:")

    print(
        f"  New: {new_output}"
    )

    print(
        f"  Old: {old_output}"
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Split sampled JSONL into new and old "
            "datasets using sample_origin."
        )
    )

    parser.add_argument(
        "--input-file",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--new-output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--old-output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--new-origin",
        default="new_stratified",
        help=(
            "sample_origin value considered new data. "
            "Default: new_stratified"
        ),
    )

    args = parser.parse_args()

    if not args.input_file.exists():

        raise FileNotFoundError(
            f"Input file not found: "
            f"{args.input_file}"
        )

    split_by_sample_origin(
        input_file=args.input_file,
        new_output=args.new_output,
        old_output=args.old_output,
        new_origin=args.new_origin,
    )


if __name__ == "__main__":
    main()