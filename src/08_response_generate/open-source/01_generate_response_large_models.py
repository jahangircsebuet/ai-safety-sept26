#!/usr/bin/env python3

# =========================================================
# IMPORTS
# =========================================================

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

from openai import OpenAI


# =========================================================
# JSON HELPERS
# =========================================================

def clean_json_value(value: Any) -> Any:
    """
    Recursively convert NaN/Inf into None so that the
    output is strict valid JSON.
    """

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    if isinstance(value, dict):
        return {
            k: clean_json_value(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            clean_json_value(v)
            for v in value
        ]

    return value


def append_to_jsonl(
    file_path: Path,
    records,
) -> None:
    """
    Append records to output JSONL.
    """

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open(
        "a",
        encoding="utf-8",
    ) as f:

        for record in records:

            record = clean_json_value(
                record
            )

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


# =========================================================
# INPUT LOADING
# =========================================================

def load_filtered_data(
    input_file: Path,
):
    """
    Load already-filtered multilingual AEGIS JSONL.

    Expected important fields:

        root_id
        language
        category
        prompt_translated

    The response output retains identity and generation metadata.
    """

    records = []

    invalid_json = 0
    missing_fields = 0
    duplicate_triples = 0

    seen_triples: Set[
        Tuple[str, str, str]
    ] = set()

    required_fields = [
        "root_id",
        "language_name",
        "category",
        "prompt_translated_lang",
    ]


    with input_file.open(
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
            # Required field check
            # ---------------------------------------------

            missing = [
                field
                for field in required_fields
                if obj.get(field) is None
            ]

            if missing:

                missing_fields += 1

                print(
                    f"[WARN] Line {line_num}: "
                    f"missing {missing}"
                )

                continue


            root_id = str(
                obj["root_id"]
            ).strip()

            language = str(
                obj["language_name"]
            ).strip()

            category = str(
                obj["category"]
            ).strip()

            prompt = str(
                obj["prompt_translated_lang"]
            ).strip()


            if not prompt:

                print(
                    f"[WARN] Empty prompt at "
                    f"line {line_num}"
                )

                continue


            # ---------------------------------------------
            # Dataset identity
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


            seen_triples.add(
                triple
            )


            records.append({

                "root_id":
                    root_id,

                "sample_id":
                    obj.get("sample_id"),

                "nllb_code":
                    obj.get("nllb_code"),

                "language":
                    language,

                "category":
                    category,

                "prompt":
                    prompt,

            })


    print("\n========================================")
    print("INPUT SUMMARY")
    print("========================================")

    print(
        f"Valid records       : "
        f"{len(records):,}"
    )

    print(
        f"Duplicate triples   : "
        f"{duplicate_triples:,}"
    )

    print(
        f"Missing fields      : "
        f"{missing_fields:,}"
    )

    print(
        f"Invalid JSON rows   : "
        f"{invalid_json:,}"
    )


    return records


# =========================================================
# RESUME SUPPORT
# =========================================================

def load_processed_keys(
    output_file: Path,
) -> set:
    """
    Load previously completed samples.

    Unique generation identity:

        (
            root_id,
            language,
            category,
            model
        )
    """

    processed = set()


    if not output_file.exists():
        return processed


    with output_file.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                obj = json.loads(line)

                root_id = obj.get(
                    "root_id"
                )

                language = obj.get(
                    "language"
                )

                category = obj.get(
                    "category"
                )

                if category is None:

                    category = (
                        obj.get(
                            "eval",
                            {}
                        ).get(
                            "category"
                        )
                    )

                model = (
                    obj.get(
                        "meta",
                        {}
                    ).get(
                        "model"
                    )
                )


                if all([
                    root_id,
                    language,
                    category,
                    model,
                ]):

                    processed.add(
                        (
                            str(root_id),
                            str(language),
                            str(category),
                            str(model),
                        )
                    )

            except Exception:
                continue


    print(
        f"Resume loaded       : "
        f"{len(processed):,} existing responses"
    )


    return processed


# =========================================================
# MINDROUTER API
# =========================================================

def call_mindrouter_api(
    client: OpenAI,
    prompt: str,
    model_name: str,
    system_instruction: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Call MindRouter's OpenAI-compatible endpoint.
    """

    start = time.time()


    try:

        response = (
            client.chat.completions.create(

                model=
                    model_name,

                messages=[
                    {
                        "role": "system",
                        "content":
                            system_instruction,
                    },
                    {
                        "role": "user",
                        "content":
                            prompt,
                    },
                ],

                temperature=
                    temperature,

                top_p=
                    top_p,

                max_tokens=
                    max_tokens,

                seed=
                    seed,
            )
        )


        latency = (
            time.time()
            - start
        )


        choice = (
            response.choices[0]
        )


        # ---------------------------------------------
        # Normal response
        # ---------------------------------------------

        content = (
            choice.message.content
        )


        # ---------------------------------------------
        # Some reasoning models may place their output
        # under reasoning_content instead.
        # ---------------------------------------------

        if content is None:

            content = getattr(
                choice.message,
                "reasoning_content",
                None,
            )


        if content is None:
            content = ""


        # ---------------------------------------------
        # Token usage
        # ---------------------------------------------

        usage = getattr(
            response,
            "usage",
            None,
        )


        prompt_tokens = (
            getattr(
                usage,
                "prompt_tokens",
                None,
            )
            if usage
            else None
        )


        completion_tokens = (
            getattr(
                usage,
                "completion_tokens",
                None,
            )
            if usage
            else None
        )


        total_tokens = (
            getattr(
                usage,
                "total_tokens",
                None,
            )
            if usage
            else None
        )


        return {

            "response":
                content,

            "error":
                None,

            "provider_response_id":
                getattr(
                    response,
                    "id",
                    None,
                ),

            "prompt_length_tokens":
                prompt_tokens,

            "response_length_tokens":
                completion_tokens,

            "total_tokens":
                total_tokens,

            "finish_reason":
                getattr(
                    choice,
                    "finish_reason",
                    None,
                ),

            "time_per_sample_sec":
                latency,
        }


    except Exception as e:

        latency = (
            time.time()
            - start
        )


        return {

            "response":
                "ERROR",

            "error":
                str(e),

            "provider_response_id":
                None,

            "prompt_length_tokens":
                None,

            "response_length_tokens":
                None,

            "total_tokens":
                None,

            "finish_reason":
                None,

            "time_per_sample_sec":
                latency,
        }


# =========================================================
# RESPONSE GENERATION
# =========================================================

def generate_responses(
    input_file: Path,
    output_file: Path,
    model_name: str,
    model_family: str,
    api_key: str,
    base_url: str,
    system_instruction: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    save_every: int,
) -> None:


    # -----------------------------------------------------
    # Load filtered benchmark
    # -----------------------------------------------------

    data = load_filtered_data(
        input_file
    )


    # -----------------------------------------------------
    # Resume
    # -----------------------------------------------------

    processed = load_processed_keys(
        output_file
    )


    # -----------------------------------------------------
    # Select remaining samples
    # -----------------------------------------------------

    pending = []


    for item in data:

        key = (
            item["root_id"],
            item["language"],
            item["category"],
            model_name,
        )


        if key not in processed:

            pending.append(
                item
            )


    print("\n========================================")
    print("GENERATION PLAN")
    print("========================================")

    print(
        f"Input file          : "
        f"{input_file}"
    )

    print(
        f"Output file         : "
        f"{output_file}"
    )

    print(
        f"Model               : "
        f"{model_name}"
    )

    print(
        f"Model family        : "
        f"{model_family}"
    )

    print(
        f"Base URL            : "
        f"{base_url}"
    )

    print(
        f"Total dataset       : "
        f"{len(data):,}"
    )

    print(
        f"Already processed   : "
        f"{len(data) - len(pending):,}"
    )

    print(
        f"Remaining           : "
        f"{len(pending):,}"
    )


    if not pending:

        print(
            "\nAll samples already processed."
        )

        return


    # -----------------------------------------------------
    # Create API client once.
    # -----------------------------------------------------

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    buffer = []

    run_start = time.time()


    # =====================================================
    # GENERATE
    # =====================================================

    for idx, item in enumerate(
        pending,
        start=1,
    ):


        result = call_mindrouter_api(

            client=
                client,

            prompt=
                item["prompt"],

            model_name=
                model_name,

            system_instruction=
                system_instruction,

            temperature=
                temperature,

            top_p=
                top_p,

            max_tokens=
                max_tokens,

            seed=
                seed,
        )


        response_text = (
            result["response"]
        )


        record = {

            # =================================================
            # Dataset identity
            # =================================================

            "root_id":
                item["root_id"],

            "sample_id":
                item["sample_id"],

            "nllb_code":
                item["nllb_code"],

            "language":
                item["language"],

            "category":
                item["category"],


            # =================================================
            # Prompt / response
            # =================================================

            "prompt":
                item["prompt"],

            "response":
                response_text,


            # =================================================
            # Model metadata
            # =================================================

            "meta": {

                "model":
                    model_name,

                "model_family":
                    model_family,

                "base_url":
                    base_url,

                "temperature":
                    temperature,

                "top_p":
                    top_p,

                "max_new_tokens":
                    max_tokens,

                "seed":
                    seed,

                "system_instruction":
                    system_instruction,

                "language":
                    item["language"],

                "prompt_length_tokens":
                    result[
                        "prompt_length_tokens"
                    ],

                "response_length_tokens":
                    result[
                        "response_length_tokens"
                    ],

                "total_tokens":
                    result[
                        "total_tokens"
                    ],

                "prompt_length_chars":
                    len(
                        item["prompt"]
                    ),

                "response_length_chars":
                    (
                        len(response_text)
                        if response_text
                        and response_text != "ERROR"
                        else 0
                    ),

                "time_per_sample_sec":
                    result[
                        "time_per_sample_sec"
                    ],

                "provider_response_id":
                    result[
                        "provider_response_id"
                    ],

                "finish_reason":
                    result[
                        "finish_reason"
                    ],

                "error":
                    result[
                        "error"
                    ],
            },


        }


        buffer.append(
            record
        )


        # -------------------------------------------------
        # Periodically append results.
        # -------------------------------------------------

        if len(buffer) >= save_every:

            append_to_jsonl(
                output_file,
                buffer,
            )


            print(
                f"Saved {len(buffer):,} rows "
                f"[{idx:,}/{len(pending):,}]"
            )


            buffer = []


    # -----------------------------------------------------
    # Flush remaining records.
    # -----------------------------------------------------

    if buffer:

        append_to_jsonl(
            output_file,
            buffer,
        )


        print(
            f"Saved final "
            f"{len(buffer):,} rows"
        )


    elapsed = (
        time.time()
        - run_start
    )


    print("\n========================================")
    print("GENERATION COMPLETE")
    print("========================================")

    print(
        f"Generated this run  : "
        f"{len(pending):,}"
    )

    print(
        f"Elapsed seconds     : "
        f"{elapsed:.2f}"
    )

    print(
        f"Output file         : "
        f"{output_file}"
    )


# =========================================================
# CLI
# =========================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate responses through MindRouter "
            "from one filtered multilingual AEGIS JSONL."
        )
    )


    parser.add_argument(
        "--input-file",
        required=True,
        type=Path,
        help=(
            "Filtered multilingual AEGIS JSONL"
        ),
    )


    parser.add_argument(
        "--output-file",
        required=True,
        type=Path,
        help=(
            "JSONL file to append responses to"
        ),
    )


    parser.add_argument(
        "--model-name",
        required=True,
        type=str,
        help=(
            "MindRouter model name"
        ),
    )


    parser.add_argument(
        "--model-family",
        required=True,
        type=str,
        help=(
            "Model family name"
        ),
    )


    parser.add_argument(
        "--api-key",
        required=True,
        type=str,
        help=(
            "MindRouter API key"
        ),
    )


    parser.add_argument(
        "--base-url",
        required=True,
        type=str,
        help=(
            "MindRouter OpenAI-compatible base URL"
        ),
    )


    parser.add_argument(
        "--system-instruction",
        required=True,
        type=str,
    )


    parser.add_argument(
        "--temperature",
        required=True,
        type=float,
    )


    parser.add_argument(
        "--top-p",
        required=True,
        type=float,
    )


    parser.add_argument(
        "--max-tokens",
        required=True,
        type=int,
    )


    parser.add_argument(
        "--seed",
        required=True,
        type=int,
    )


    parser.add_argument(
        "--save-every",
        required=True,
        type=int,
        help=(
            "Append results after this many samples"
        ),
    )


    args = parser.parse_args()


    if not args.input_file.exists():

        raise FileNotFoundError(
            f"Input file not found: "
            f"{args.input_file}"
        )


    if args.save_every <= 0:

        raise ValueError(
            "--save-every must be > 0"
        )


    generate_responses(

        input_file=
            args.input_file,

        output_file=
            args.output_file,

        model_name=
            args.model_name,

        model_family=
            args.model_family,

        api_key=
            args.api_key,

        base_url=
            args.base_url,

        system_instruction=
            args.system_instruction,

        temperature=
            args.temperature,

        top_p=
            args.top_p,

        max_tokens=
            args.max_tokens,

        seed=
            args.seed,

        save_every=
            args.save_every,
    )


if __name__ == "__main__":
    main()