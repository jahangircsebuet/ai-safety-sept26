#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple


# =========================================================
# EXPECTED CURRENT RESPONSE SCHEMA
# =========================================================

EXPECTED_KEYS = [
    # -----------------------------------------------------
    # Model / request identity
    # -----------------------------------------------------
    "custom_id",
    "provider",
    "model",

    # -----------------------------------------------------
    # Original 10K benchmark metadata
    # -----------------------------------------------------
    "root_id",
    "language_name",
    "nllb_code",

    "prompt_translated_lang",
    "prompt_back_to_original_lang",

    "translation_model",

    "bertscore_f1",
    "comet_supported",
    "comet_score",
    "chrf_score",

    "sample_id",
    "category",

    "sample_origin",
    "old_source_root_ids",

    "selection_percentile",
    "sampling_strategy",

    # -----------------------------------------------------
    # Generation configuration
    # -----------------------------------------------------
    "system_instruction",
    "generation_config",

    # -----------------------------------------------------
    # Generated response
    # -----------------------------------------------------
    "response",
    "response_raw",

    "provider_response_id",

    "prompt_length_tokens",
    "response_length_tokens",
    "token_count_method",

    "time_per_sample_sec",
    "finish_reason",
    "error",

    "source_request_file",
]


# Fields that should normally exist in every generated response.
IDENTITY_FIELDS = [
    "custom_id",
    "provider",
    "model",
    "sample_id",
    "root_id",
    "language_name",
    "category",
    "prompt_translated_lang",
]


# =========================================================
# JSON HELPERS
# =========================================================

def clean_json_value(value: Any) -> Any:
    """
    Recursively convert non-finite float values such as:

        NaN
        +Inf
        -Inf

    into None so the final JSONL remains standards-compliant.
    """

    if isinstance(value, float):

        if not math.isfinite(value):
            return None

        return value

    if isinstance(value, list):

        return [
            clean_json_value(v)
            for v in value
        ]

    if isinstance(value, dict):

        return {
            key: clean_json_value(val)
            for key, val in value.items()
        }

    return value


def iter_jsonl(
    path: Path,
) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """
    Stream one JSON object at a time.

    This avoids loading an entire model response file into RAM,
    which is useful because response_raw can make files large.
    """

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line_num, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                row = json.loads(line)

            except json.JSONDecodeError as e:

                print(
                    f"[WARN] Invalid JSON skipped: "
                    f"{path}:{line_num}: {e}"
                )

                continue

            if not isinstance(row, dict):

                print(
                    f"[WARN] Non-object JSON skipped: "
                    f"{path}:{line_num}"
                )

                continue

            yield line_num, row


def write_jsonl_row(
    fout,
    row: Dict[str, Any],
) -> None:
    """
    Write one JSONL row.
    """

    row = clean_json_value(row)

    fout.write(
        json.dumps(
            row,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


# =========================================================
# SCHEMA HARMONIZATION
# =========================================================

def harmonize_row(
    row: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Preserve ALL existing fields.

    Only add expected fields that are missing.

    No fields are removed.
    """

    out = dict(row)

    for key in EXPECTED_KEYS:

        if key not in out:

            if key == "generation_config":

                out[key] = {}

            elif key == "old_source_root_ids":

                out[key] = []

            else:

                out[key] = None

    return out


# =========================================================
# IDENTITY VALIDATION
# =========================================================

def find_missing_identity_fields(
    row: Dict[str, Any],
) -> list[str]:
    """
    Return important identifying fields that are absent or empty.
    """

    missing = []

    for field in IDENTITY_FIELDS:

        value = row.get(field)

        if value is None:
            missing.append(field)

        elif isinstance(value, str) and not value.strip():
            missing.append(field)

    return missing


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Merge closed-source LLM response JSONL files "
            "while preserving the complete 10K benchmark metadata."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help=(
            "Directory containing model response JSONL files. "
            "Subdirectories are searched recursively."
        ),
    )

    parser.add_argument(
        "--out-file",
        required=True,
        type=Path,
        help="Merged response JSONL filepath.",
    )

    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help=(
            "Skip rows whose error field is non-empty. "
            "For benchmark auditing, usually leave this OFF."
        ),
    )

    parser.add_argument(
        "--deduplicate-by-custom-id",
        action="store_true",
        help=(
            "Keep only the first occurrence of each custom_id."
        ),
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    out_file = args.out_file

    # =====================================================
    # VALIDATION
    # =====================================================

    if not input_dir.exists():

        raise FileNotFoundError(
            f"Input directory does not exist: "
            f"{input_dir}"
        )

    if not input_dir.is_dir():

        raise NotADirectoryError(
            f"Input path is not a directory: "
            f"{input_dir}"
        )

    out_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Find response files recursively.
    #
    # Example:
    #
    # 08_llm_responses/
    # ├── anthropic/
    # │   └── claude_haiku_4_5.jsonl
    # ├── openai/
    # │   └── gpt_5_4_mini.jsonl
    # └── gemini/
    #     └── gemini_2_5_flash.jsonl
    # -----------------------------------------------------

    files = sorted(
        f
        for f in input_dir.rglob("*.jsonl")
        if f.is_file()
        and f.resolve() != out_file.resolve()
    )

    if not files:

        raise FileNotFoundError(
            f"No JSONL response files found under: "
            f"{input_dir}"
        )

    print()
    print("========================================")
    print("FILES TO MERGE")
    print("========================================")

    for file in files:
        print(file)

    print(
        f"\nTotal files: {len(files):,}"
    )

    # =====================================================
    # COUNTERS
    # =====================================================

    seen_custom_ids = set()

    total_rows = 0
    written_rows = 0

    skipped_errors = 0
    skipped_duplicates = 0

    invalid_identity_rows = 0

    provider_counts = Counter()
    model_counts = Counter()
    provider_model_counts = Counter()

    error_counts_by_model = Counter()

    # =====================================================
    # WRITE OUTPUT
    # =====================================================

    # Always rebuild the merged output from scratch.
    if out_file.exists():

        print(
            f"\nRemoving previous merged output: "
            f"{out_file}"
        )

        out_file.unlink()

    with open(
        out_file,
        "w",
        encoding="utf-8",
    ) as fout:

        for file in files:

            print()
            print(
                f"Loading response file: {file}"
            )

            file_seen = 0
            file_written = 0

            for line_num, row in iter_jsonl(
                file
            ):

                total_rows += 1
                file_seen += 1

                # -----------------------------------------
                # Identity validation
                # -----------------------------------------

                missing_identity = (
                    find_missing_identity_fields(
                        row
                    )
                )

                if missing_identity:

                    invalid_identity_rows += 1

                    print(
                        f"[WARN] Missing identity fields "
                        f"in {file.name}:{line_num}: "
                        f"{missing_identity}"
                    )

                    # Do NOT drop the row.
                    # Preserve it for auditing.

                # -----------------------------------------
                # Count API/model errors
                # -----------------------------------------

                error_value = row.get(
                    "error"
                )

                has_error = (
                    error_value
                    not in (
                        None,
                        "",
                        {},
                        [],
                    )
                )

                if has_error:

                    error_counts_by_model[
                        row.get(
                            "model",
                            "<missing-model>",
                        )
                    ] += 1

                # -----------------------------------------
                # Optional error filtering
                # -----------------------------------------

                if (
                    args.skip_errors
                    and has_error
                ):

                    skipped_errors += 1
                    continue

                # -----------------------------------------
                # Optional deduplication
                # -----------------------------------------

                cid = row.get(
                    "custom_id"
                )

                if (
                    args.deduplicate_by_custom_id
                    and cid is not None
                ):

                    if cid in seen_custom_ids:

                        skipped_duplicates += 1

                        print(
                            f"[WARN] Duplicate custom_id "
                            f"skipped: {cid}"
                        )

                        continue

                    seen_custom_ids.add(
                        cid
                    )

                # -----------------------------------------
                # Harmonize schema
                # -----------------------------------------

                out_row = harmonize_row(
                    row
                )

                # -----------------------------------------
                # Write immediately
                # -----------------------------------------

                write_jsonl_row(
                    fout,
                    out_row,
                )

                written_rows += 1
                file_written += 1

                # -----------------------------------------
                # Statistics
                # -----------------------------------------

                provider = (
                    out_row.get("provider")
                    or "<missing-provider>"
                )

                model = (
                    out_row.get("model")
                    or "<missing-model>"
                )

                provider_counts[
                    provider
                ] += 1

                model_counts[
                    model
                ] += 1

                provider_model_counts[
                    (
                        provider,
                        model,
                    )
                ] += 1

            print(
                f"  Rows seen    : "
                f"{file_seen:,}"
            )

            print(
                f"  Rows written : "
                f"{file_written:,}"
            )

    # =====================================================
    # FINAL REPORT
    # =====================================================

    print()
    print("========================================")
    print("MERGE COMPLETE")
    print("========================================")

    print(
        f"Response files        : "
        f"{len(files):,}"
    )

    print(
        f"Rows seen             : "
        f"{total_rows:,}"
    )

    print(
        f"Rows written          : "
        f"{written_rows:,}"
    )

    print(
        f"Skipped errors        : "
        f"{skipped_errors:,}"
    )

    print(
        f"Skipped duplicates    : "
        f"{skipped_duplicates:,}"
    )

    print(
        f"Rows missing identity : "
        f"{invalid_identity_rows:,}"
    )

    # -----------------------------------------------------
    # Provider/model counts
    # -----------------------------------------------------

    print()
    print("========================================")
    print("ROWS BY PROVIDER / MODEL")
    print("========================================")

    for (
        provider,
        model,
    ), count in sorted(
        provider_model_counts.items()
    ):

        print(
            f"{provider:12s} | "
            f"{model:35s} | "
            f"{count:,}"
        )

    # -----------------------------------------------------
    # Errors by model
    # -----------------------------------------------------

    print()
    print("========================================")
    print("ERRORS BY MODEL")
    print("========================================")

    if error_counts_by_model:

        for model, count in sorted(
            error_counts_by_model.items()
        ):

            print(
                f"{model:35s} | "
                f"{count:,}"
            )

    else:

        print(
            "No error rows found."
        )

    # -----------------------------------------------------
    # Expected benchmark sanity check
    # -----------------------------------------------------

    print()
    print("========================================")
    print("SAMPLE-ID COVERAGE BY MODEL")
    print("========================================")

    # Read merged file one more time and count unique sample_ids/model.
    samples_by_model = {}

    for _, row in iter_jsonl(
        out_file
    ):

        model = row.get(
            "model"
        )

        sample_id = row.get(
            "sample_id"
        )

        if model is None or sample_id is None:
            continue

        samples_by_model.setdefault(
            model,
            set(),
        ).add(
            sample_id
        )

    for model in sorted(
        samples_by_model
    ):

        unique_count = len(
            samples_by_model[
                model
            ]
        )

        print(
            f"{model:35s} | "
            f"{unique_count:,} unique sample_ids"
        )

    print()
    print(
        f"Merged output saved to:"
        f"\n{out_file}"
    )


if __name__ == "__main__":
    main()