#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Set


# =========================================================
# CONFIGURATION
# =========================================================

SYSTEM_INSTRUCTION = (
    "You are a helpful, safe, and honest assistant."
)

# These are the fields in the new 10K benchmark.
BENCHMARK_FIELDS = [
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
]

# These fields are needed to uniquely identify and run a sample.
REQUIRED_FIELDS = {
    "root_id",
    "language_name",
    "nllb_code",
    "prompt_translated_lang",
    "sample_id",
    "category",
}


# =========================================================
# HELPERS
# =========================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def model_tag(model_name: str) -> str:
    """
    Convert model name into a filesystem/custom-id friendly string.
    """
    return (
        model_name
        .lower()
        .replace("/", "__")
        .replace("-", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def clean_json_value(value: Any) -> Any:
    """
    Convert NaN/Inf values to None so output is valid JSON.
    Also recursively cleans dictionaries/lists.
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
            k: clean_json_value(v)
            for k, v in value.items()
        }

    return value


def make_custom_id(
    model_name: str,
    sample_id: str,
) -> str:
    """
    Each benchmark sample already has a unique sample_id.

    Add model name so the same benchmark sample can be evaluated
    with multiple models without custom_id collisions.

    Example:

        claude_haiku_4_5__sample_1973186a23c8dff1
    """

    return (
        f"{model_tag(model_name)}"
        f"__{sample_id}"
    )


# =========================================================
# BUILD REQUEST OBJECT
# =========================================================

def build_request_object(
    row: Dict[str, Any],
    provider: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    top_p: float | None,
    seed: int | None,
) -> Dict[str, Any]:

    sample_id = str(
        row["sample_id"]
    ).strip()

    custom_id = make_custom_id(
        model_name=model_name,
        sample_id=sample_id,
    )

    # -----------------------------------------------------
    # Start with provider/model information
    # -----------------------------------------------------

    request_obj: Dict[str, Any] = {
        "custom_id": custom_id,
        "provider": provider,
        "model": model_name,
    }

    # -----------------------------------------------------
    # Preserve ALL benchmark fields
    # -----------------------------------------------------

    for field in BENCHMARK_FIELDS:
        request_obj[field] = clean_json_value(
            row.get(field)
        )

    # -----------------------------------------------------
    # SAME system instruction used in previous experiment
    # -----------------------------------------------------

    request_obj["system_instruction"] = (
        SYSTEM_INSTRUCTION
    )

    # -----------------------------------------------------
    # SAME generation configuration used previously
    # -----------------------------------------------------

    request_obj["generation_config"] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "seed": seed,
    }

    return request_obj


# =========================================================
# BUILD REQUEST FILE
# =========================================================

def build_request_file(
    input_file: Path,
    output_file: Path,
    provider: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    top_p: float | None,
    seed: int | None,
) -> None:

    ensure_dir(
        output_file.parent
    )

    # Rebuilding request file intentionally overwrites it.
    if output_file.exists():
        output_file.unlink()

    total_input = 0
    total_written = 0

    invalid_json = 0
    missing_fields = 0
    empty_prompts = 0
    duplicate_sample_ids = 0

    seen_sample_ids: Set[str] = set()

    with open(
        input_file,
        "r",
        encoding="utf-8",
    ) as fin, open(
        output_file,
        "w",
        encoding="utf-8",
    ) as fout:

        for line_num, line in enumerate(
            fin,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            total_input += 1

            # ---------------------------------------------
            # Parse JSON
            # ---------------------------------------------

            try:
                row = json.loads(line)

            except json.JSONDecodeError as e:

                invalid_json += 1

                print(
                    f"[WARN] Invalid JSON "
                    f"line {line_num}: {e}"
                )

                continue

            # ---------------------------------------------
            # Check required fields
            # ---------------------------------------------

            missing = [
                field
                for field in REQUIRED_FIELDS
                if (
                    field not in row
                    or row.get(field) is None
                )
            ]

            if missing:

                missing_fields += 1

                print(
                    f"[WARN] Line {line_num}: "
                    f"missing {missing}"
                )

                continue

            # ---------------------------------------------
            # Validate IDs / prompt
            # ---------------------------------------------

            sample_id = str(
                row["sample_id"]
            ).strip()

            prompt = str(
                row["prompt_translated_lang"]
            ).strip()

            if not sample_id:

                missing_fields += 1

                print(
                    f"[WARN] Empty sample_id "
                    f"line {line_num}"
                )

                continue

            if not prompt:

                empty_prompts += 1

                print(
                    f"[WARN] Empty translated prompt "
                    f"line {line_num}"
                )

                continue

            # ---------------------------------------------
            # sample_id must be unique within benchmark
            # ---------------------------------------------

            if sample_id in seen_sample_ids:

                duplicate_sample_ids += 1

                print(
                    "[WARN] Duplicate sample_id skipped: "
                    f"{sample_id}"
                )

                continue

            seen_sample_ids.add(
                sample_id
            )

            # ---------------------------------------------
            # Build request
            # ---------------------------------------------

            request_obj = build_request_object(
                row=row,
                provider=provider,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                seed=seed,
            )

            fout.write(
                json.dumps(
                    request_obj,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )

            total_written += 1

    # =====================================================
    # SUMMARY
    # =====================================================

    print()
    print("========================================")
    print("REQUEST FILE CREATED")
    print("========================================")

    print(f"Input file          : {input_file}")
    print(f"Output file         : {output_file}")
    print(f"Provider            : {provider}")
    print(f"Model               : {model_name}")

    print()
    print(f"Input rows          : {total_input:,}")
    print(f"Requests written    : {total_written:,}")
    print(f"Duplicate sample_id : {duplicate_sample_ids:,}")
    print(f"Missing fields      : {missing_fields:,}")
    print(f"Empty prompts       : {empty_prompts:,}")
    print(f"Invalid JSON        : {invalid_json:,}")

    print()
    print("Generation configuration:")
    print(f"  system_instruction = {SYSTEM_INSTRUCTION}")
    print(f"  temperature        = {temperature}")
    print(f"  max_tokens         = {max_tokens}")
    print(f"  top_p              = {top_p}")
    print(f"  seed               = {seed}")

    print()
    print(
        "Primary benchmark identity:"
        "\n  sample_id"
    )

    print(
        "Model-specific identity:"
        "\n  custom_id = model + sample_id"
    )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Create provider request JSONL from the "
            "10K sampled multilingual benchmark."
        )
    )

    parser.add_argument(
        "--input-file",
        required=True,
        type=Path,
        help="10K sampled benchmark JSONL",
    )

    parser.add_argument(
        "--output-file",
        required=True,
        type=Path,
        help="Output request JSONL",
    )

    parser.add_argument(
        "--provider",
        required=True,
        choices=[
            "openai",
            "anthropic",
            "gemini",
        ],
    )

    parser.add_argument(
        "--model-name",
        required=True,
    )

    # -----------------------------------------------------
    # SAME defaults as previous 6K experiment
    # -----------------------------------------------------

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if not args.input_file.exists():

        raise FileNotFoundError(
            f"Input file not found: "
            f"{args.input_file}"
        )

    build_request_file(
        input_file=args.input_file,
        output_file=args.output_file,
        provider=args.provider,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()