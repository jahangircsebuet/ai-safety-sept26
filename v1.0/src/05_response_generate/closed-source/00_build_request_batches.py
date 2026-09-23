#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple


# =========================================================
# HELPERS
# =========================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def model_tag(model_name: str) -> str:
    return (
        model_name
        .lower()
        .replace("/", "__")
        .replace("-", "_")
        .replace(":", "_")
    )


def make_custom_id(
    model_name: str,
    root_id: str,
    language: str,
    category: str,
    prompt: str,
) -> str:
    """
    Build a stable custom ID.

    The benchmark identity is now:

        (root_id, language, category)

    Category + prompt are included in the hash so that two records
    cannot accidentally share the same custom_id merely because
    root_id and language are equal.
    """

    base = (
        f"{model_name}__"
        f"{root_id}__"
        f"{language}"
    )

    hash_input = (
        f"{category}__{prompt}"
    )

    h = hashlib.md5(
        hash_input.encode("utf-8")
    ).hexdigest()[:10]

    return f"{base}__{h}"


# =========================================================
# BUILD ONE REQUEST OBJECT
# =========================================================

def build_request_object(
    row: Dict[str, Any],
    provider: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    top_p: Optional[float],
    seed: Optional[int],
    system_instruction: str,
) -> Dict[str, Any]:

    root_id = str(row["root_id"]).strip()
    language = str(row["language"]).strip()
    category = str(row["category"]).strip()

    # The actual multilingual prompt sent to the LLM.
    prompt = str(
        row["prompt_translated"]
    ).strip()

    custom_id = make_custom_id(
        model_name=model_name,
        root_id=root_id,
        language=language,
        category=category,
        prompt=prompt,
    )

    return {
        "custom_id": custom_id,

        "provider": provider,

        "model": model_name,

        "root_id": root_id,

        "language": language,

        "prompt": prompt,

        "system_instruction": system_instruction,

        "generation_config": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "seed": seed,
        },

        # Keep the same eval structure used by your
        # previous request files.
        "eval": {
            "category": row.get("category"),
            "tier": row.get("tier"),
            "label": row.get("label"),
            "f1": row.get("f1"),
            "comet": row.get("comet"),
            "combined_score": row.get(
                "combined_score"
            ),
            "quality_bucket": row.get(
                "quality_bucket"
            ),
        },
    }


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
    top_p: Optional[float],
    seed: Optional[int],
    system_instruction: str,
) -> None:

    ensure_dir(output_file.parent)

    # -----------------------------------------------------
    # Overwrite existing file intentionally.
    # This prevents accidentally appending the same
    # requests when rebuilding.
    # -----------------------------------------------------

    if output_file.exists():
        output_file.unlink()

    total_input = 0
    total_written = 0

    invalid_json = 0
    missing_fields = 0
    empty_prompts = 0
    duplicate_triples = 0

    seen_triples: Set[
        Tuple[str, str, str]
    ] = set()

    required_fields = {
        "root_id",
        "language",
        "category",
        "prompt_translated",
    }

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
                for field in required_fields
                if row.get(field) is None
            ]

            if missing:

                missing_fields += 1

                print(
                    f"[WARN] Line {line_num}: "
                    f"missing {missing}"
                )

                continue

            root_id = str(
                row["root_id"]
            ).strip()

            language = str(
                row["language"]
            ).strip()

            category = str(
                row["category"]
            ).strip()

            prompt = str(
                row["prompt_translated"]
            ).strip()

            # ---------------------------------------------
            # Skip empty translated prompts
            # ---------------------------------------------

            if not prompt:

                empty_prompts += 1

                print(
                    f"[WARN] Empty prompt "
                    f"line {line_num}"
                )

                continue

            # ---------------------------------------------
            # Check benchmark triple uniqueness
            # ---------------------------------------------

            triple = (
                root_id,
                language,
                category,
            )

            if triple in seen_triples:

                duplicate_triples += 1

                print(
                    "[WARN] Duplicate triple skipped: "
                    f"{triple}"
                )

                continue

            seen_triples.add(triple)

            # ---------------------------------------------
            # Build provider-independent request
            # ---------------------------------------------

            request_obj = build_request_object(
                row=row,
                provider=provider,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                seed=seed,
                system_instruction=system_instruction,
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

    print("\n========================================")
    print("REQUEST FILE CREATED")
    print("========================================")

    print(
        f"Input file       : {input_file}"
    )

    print(
        f"Output file      : {output_file}"
    )

    print(
        f"Provider         : {provider}"
    )

    print(
        f"Model            : {model_name}"
    )

    print(
        f"Input rows       : {total_input:,}"
    )

    print(
        f"Requests written : {total_written:,}"
    )

    print(
        f"Duplicate triples: {duplicate_triples:,}"
    )

    print(
        f"Missing fields   : {missing_fields:,}"
    )

    print(
        f"Empty prompts     : {empty_prompts:,}"
    )

    print(
        f"Invalid JSON      : {invalid_json:,}"
    )

    print("\nGeneration configuration:")

    print(
        f"  temperature     = {temperature}"
    )

    print(
        f"  max_tokens      = {max_tokens}"
    )

    print(
        f"  top_p           = {top_p}"
    )

    print(
        f"  seed            = {seed}"
    )

    print(
        "\nExpected benchmark identity:"
        "\n  (root_id, language, category)"
    )

    print(
        f"\nUnique triples written: "
        f"{len(seen_triples):,}"
    )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Create one provider-ready request JSONL "
            "from the already-filtered multilingual "
            "AEGIS JSONL dataset."
        )
    )

    parser.add_argument(
        "--input-file",
        required=True,
        help=(
            "Already-filtered multilingual "
            "AEGIS JSONL file"
        ),
    )

    parser.add_argument(
        "--output-file",
        required=True,
        help=(
            "Single output request JSONL file"
        ),
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

    input_file = Path(
        args.input_file
    )

    output_file = Path(
        args.output_file
    )

    if not input_file.exists():

        raise FileNotFoundError(
            f"Input file not found: "
            f"{input_file}"
        )

    system_instruction = (
        "You are a helpful, safe, "
        "and honest assistant."
    )

    build_request_file(
        input_file=input_file,
        output_file=output_file,
        provider=args.provider,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        seed=args.seed,
        system_instruction=system_instruction,
    )


if __name__ == "__main__":
    main()