#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
import anthropic

from google import genai
from google.genai import types as genai_types

import tiktoken


# =========================================================
# TOKENIZER FALLBACK
# =========================================================

ENCODING = tiktoken.get_encoding(
    "cl100k_base"
)


# =========================================================
# HELPERS
# =========================================================

def make_serializable(obj: Any) -> Any:

    if isinstance(obj, dict):
        return {
            k: make_serializable(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [
            make_serializable(v)
            for v in obj
        ]

    if isinstance(
        obj,
        (
            str,
            int,
            float,
            bool,
            type(None),
        ),
    ):
        return obj

    try:
        return obj.model_dump()

    except AttributeError:
        return str(obj)


def ensure_dir(path: Path) -> None:

    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_jsonl(
    path: Path,
) -> List[Dict[str, Any]]:

    rows: List[
        Dict[str, Any]
    ] = []

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

                rows.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as e:

                print(
                    f"[WARN] Invalid JSON "
                    f"{path.name}:{line_num}: {e}"
                )

    return rows


def append_jsonl(
    path: Path,
    records: List[Dict[str, Any]],
) -> None:

    ensure_dir(
        path.parent
    )

    with open(
        path,
        "a",
        encoding="utf-8",
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def approx_token_count(
    text: Optional[str],
) -> Optional[int]:

    if text is None:
        return None

    try:

        return len(
            ENCODING.encode(text)
        )

    except Exception:

        return max(
            1,
            len(text) // 4,
        )


# =========================================================
# RESUME
# =========================================================

def load_processed_ids(
    output_file: Path,
) -> set:

    processed = set()

    if not output_file.exists():
        return processed

    for row in load_jsonl(
        output_file
    ):

        cid = row.get(
            "custom_id"
        )

        if cid:
            processed.add(cid)

    print(
        f"Resume loaded: "
        f"{len(processed):,} completed requests"
    )

    return processed


# =========================================================
# PROVIDER USAGE
# =========================================================

def extract_openai_usage(
    resp: Any,
    prompt: str,
    output_text: str,
) -> Tuple[
    Optional[int],
    Optional[int],
    str,
]:

    usage = getattr(
        resp,
        "usage",
        None,
    )

    if usage is not None:

        in_tok = getattr(
            usage,
            "input_tokens",
            None,
        )

        out_tok = getattr(
            usage,
            "output_tokens",
            None,
        )

        if (
            in_tok is not None
            or out_tok is not None
        ):

            return (
                in_tok,
                out_tok,
                "provider_reported_or_partial",
            )

    return (
        approx_token_count(prompt),
        approx_token_count(output_text),
        "tiktoken_cl100k_fallback",
    )


def extract_anthropic_usage(
    resp: Any,
    prompt: str,
    output_text: str,
) -> Tuple[
    Optional[int],
    Optional[int],
    str,
]:

    usage = getattr(
        resp,
        "usage",
        None,
    )

    if usage is not None:

        in_tok = getattr(
            usage,
            "input_tokens",
            None,
        )

        out_tok = getattr(
            usage,
            "output_tokens",
            None,
        )

        if (
            in_tok is not None
            or out_tok is not None
        ):

            return (
                in_tok,
                out_tok,
                "provider_reported_or_partial",
            )

    return (
        approx_token_count(prompt),
        approx_token_count(output_text),
        "tiktoken_cl100k_fallback",
    )


def extract_gemini_usage(
    resp: Any,
    prompt: str,
    output_text: str,
) -> Tuple[
    Optional[int],
    Optional[int],
    str,
]:

    usage = getattr(
        resp,
        "usage_metadata",
        None,
    )

    if usage is not None:

        in_tok = getattr(
            usage,
            "prompt_token_count",
            None,
        )

        out_tok = getattr(
            usage,
            "candidates_token_count",
            None,
        )

        if (
            in_tok is not None
            or out_tok is not None
        ):

            return (
                in_tok,
                out_tok,
                "provider_reported_or_partial",
            )

    return (
        approx_token_count(prompt),
        approx_token_count(output_text),
        "tiktoken_cl100k_fallback",
    )


# =========================================================
# CLIENTS
# =========================================================

def get_openai_client(api_key: str) -> OpenAI:

    if not api_key:

        raise RuntimeError(
            "--openai-api-key is required for OpenAI requests"
        )

    return OpenAI(
        api_key=api_key
    )


def get_anthropic_client(api_key: str) -> anthropic.Anthropic:

    if not api_key:

        raise RuntimeError(
            "--anthropic-api-key is required for Anthropic requests"
        )

    return anthropic.Anthropic(
        api_key=api_key
    )


def get_gemini_client(api_key: str) -> genai.Client:

    if not api_key:

        raise RuntimeError(
            "--gemini-api-key is required for Gemini requests"
        )

    return genai.Client(
        api_key=api_key
    )


# =========================================================
# OPENAI
# =========================================================

def generate_openai(
    request_obj: Dict[str, Any],
    client: OpenAI,
) -> Dict[str, Any]:

    prompt = request_obj[
        "prompt_translated_lang"
    ]

    cfg = request_obj[
        "generation_config"
    ]

    system_instruction = request_obj[
        "system_instruction"
    ]

    start = time.time()

    # IMPORTANT:
    # Keep same generation behavior as previous experiment.
    # top_p and seed are stored but are not passed here,
    # matching the previous runner.
    resp = client.responses.create(

        model=request_obj[
            "model"
        ],

        input=[
            {
                "role": "system",
                "content": system_instruction,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        temperature=cfg[
            "temperature"
        ],

        max_output_tokens=cfg[
            "max_tokens"
        ],
    )

    latency = (
        time.time()
        - start
    )

    output_text = getattr(
        resp,
        "output_text",
        None,
    )

    if not output_text:

        try:

            output_text = (
                resp.output[0]
                .content[0]
                .text
            )

        except Exception:

            output_text = ""

    (
        in_tok,
        out_tok,
        token_method,
    ) = extract_openai_usage(
        resp,
        prompt,
        output_text,
    )

    return {
        "provider_response_id":
            getattr(
                resp,
                "id",
                None,
            ),

        "response":
            output_text,

        "response_raw":
            make_serializable(
                resp
            ),

        "prompt_length_tokens":
            in_tok,

        "response_length_tokens":
            out_tok,

        "token_count_method":
            token_method,

        "time_per_sample_sec":
            latency,

        "finish_reason":
            None,

        "error":
            None,
    }


# =========================================================
# ANTHROPIC
# =========================================================

def generate_anthropic(
    request_obj: Dict[str, Any],
    client: anthropic.Anthropic,
) -> Dict[str, Any]:

    prompt = request_obj[
        "prompt_translated_lang"
    ]

    cfg = request_obj[
        "generation_config"
    ]

    system_instruction = request_obj[
        "system_instruction"
    ]

    start = time.time()

    # Keep same Anthropic generation configuration
    # as previous experiment.
    resp = client.messages.create(

        model=request_obj[
            "model"
        ],

        system=
            system_instruction,

        max_tokens=
            cfg["max_tokens"],

        extra_body={"temperature": cfg["temperature"]},

        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    latency = (
        time.time()
        - start
    )

    chunks = []

    for block in getattr(
        resp,
        "content",
        [],
    ):

        txt = getattr(
            block,
            "text",
            None,
        )

        if txt:

            chunks.append(
                txt
            )

    output_text = "".join(
        chunks
    )

    (
        in_tok,
        out_tok,
        token_method,
    ) = extract_anthropic_usage(
        resp,
        prompt,
        output_text,
    )

    return {
        "provider_response_id":
            getattr(
                resp,
                "id",
                None,
            ),

        "response":
            output_text,

        "response_raw":
            make_serializable(
                resp
            ),

        "prompt_length_tokens":
            in_tok,

        "response_length_tokens":
            out_tok,

        "token_count_method":
            token_method,

        "time_per_sample_sec":
            latency,

        "finish_reason":
            getattr(
                resp,
                "stop_reason",
                None,
            ),

        "error":
            None,
    }


# =========================================================
# GEMINI
# =========================================================

def generate_gemini(
    request_obj: Dict[str, Any],
    client: genai.Client,
) -> Dict[str, Any]:

    prompt = request_obj[
        "prompt_translated_lang"
    ]

    cfg = request_obj[
        "generation_config"
    ]

    system_instruction = request_obj[
        "system_instruction"
    ]

    start = time.time()

    resp = client.models.generate_content(

        model=request_obj[
            "model"
        ],

        contents=
            prompt,

        config=
            genai_types.GenerateContentConfig(

                system_instruction=
                    system_instruction,

                temperature=
                    cfg["temperature"],

                max_output_tokens=
                    cfg["max_tokens"],

                top_p=
                    cfg.get(
                        "top_p"
                    ),

                seed=
                    cfg.get(
                        "seed"
                    ),
            ),
    )

    latency = (
        time.time()
        - start
    )

    output_text = (
        getattr(
            resp,
            "text",
            "",
        )
        or ""
    )

    (
        in_tok,
        out_tok,
        token_method,
    ) = extract_gemini_usage(
        resp,
        prompt,
        output_text,
    )

    finish_reason = None

    try:

        candidates = getattr(
            resp,
            "candidates",
            None,
        )

        if candidates:

            finish_reason = getattr(
                candidates[0],
                "finish_reason",
                None,
            )

    except Exception:
        pass

    return {
        "provider_response_id":
            None,

        "response":
            output_text,

        "response_raw":
            make_serializable(
                resp
            ),

        "prompt_length_tokens":
            in_tok,

        "response_length_tokens":
            out_tok,

        "token_count_method":
            token_method,

        "time_per_sample_sec":
            latency,

        "finish_reason":
            make_serializable(
                finish_reason
            ),

        "error":
            None,
    }


# =========================================================
# ROUTER
# =========================================================

def route_generate(
    request_obj: Dict[str, Any],
    clients: Dict[str, Any],
) -> Dict[str, Any]:

    provider = request_obj[
        "provider"
    ]

    try:

        if provider == "openai":

            return generate_openai(
                request_obj,
                clients["openai"],
            )

        elif provider == "anthropic":

            return generate_anthropic(
                request_obj,
                clients["anthropic"],
            )

        elif provider == "gemini":

            return generate_gemini(
                request_obj,
                clients["gemini"],
            )

        raise ValueError(
            f"Unsupported provider: "
            f"{provider}"
        )

    except Exception as e:

        return {
            "provider_response_id":
                None,

            "response":
                None,

            "response_raw":
                None,

            "prompt_length_tokens":
                None,

            "response_length_tokens":
                None,

            "token_count_method":
                None,

            "time_per_sample_sec":
                None,

            "finish_reason":
                None,

            "error":
                str(e),
        }


# =========================================================
# PROCESS REQUEST FILE
# =========================================================

def process_request_file(
    request_file: Path,
    output_file: Path,
    clients: Dict[str, Any],
) -> None:

    print()
    print("========================================")
    print("PROCESS REQUEST FILE")
    print("========================================")

    print(
        f"Request file : {request_file}"
    )

    print(
        f"Output file  : {output_file}"
    )

    requests = load_jsonl(
        request_file
    )

    processed = load_processed_ids(
        output_file
    )

    pending = [
        request
        for request in requests
        if request.get(
            "custom_id"
        ) not in processed
    ]

    print(
        f"Total requests     : "
        f"{len(requests):,}"
    )

    print(
        f"Already completed  : "
        f"{len(processed):,}"
    )

    print(
        f"Remaining requests : "
        f"{len(pending):,}"
    )

    if not pending:

        print(
            "Nothing left to process."
        )

        return

    file_start = time.time()

    results_buffer: List[
        Dict[str, Any]
    ] = []

    for idx, req in enumerate(
        pending,
        start=1,
    ):

        gen = route_generate(
            req,
            clients,
        )

        # -------------------------------------------------
        # Preserve the ENTIRE request object.
        #
        # Therefore sample_id + all 10K benchmark metadata
        # automatically remain available in response rows.
        # -------------------------------------------------

        out_row = dict(
            req
        )

        # Add model-generation results.
        out_row.update(
            {
                "response":
                    gen.get(
                        "response"
                    ),

                "response_raw":
                    gen.get(
                        "response_raw"
                    ),

                "provider_response_id":
                    gen.get(
                        "provider_response_id"
                    ),

                "prompt_length_tokens":
                    gen.get(
                        "prompt_length_tokens"
                    ),

                "response_length_tokens":
                    gen.get(
                        "response_length_tokens"
                    ),

                "token_count_method":
                    gen.get(
                        "token_count_method"
                    ),

                "time_per_sample_sec":
                    gen.get(
                        "time_per_sample_sec"
                    ),

                "finish_reason":
                    gen.get(
                        "finish_reason"
                    ),

                "error":
                    gen.get(
                        "error"
                    ),

                "source_request_file":
                    str(
                        request_file
                    ),
            }
        )

        results_buffer.append(
            out_row
        )

        # Preserve previous incremental-saving behavior.
        if len(
            results_buffer
        ) >= 25:

            append_jsonl(
                output_file,
                results_buffer,
            )

            print(
                f"Saved {len(results_buffer)} rows "
                f"[{idx:,}/{len(pending):,}]"
            )

            results_buffer = []

    if results_buffer:

        append_jsonl(
            output_file,
            results_buffer,
        )

        print(
            f"Saved final "
            f"{len(results_buffer)} rows"
        )

    total_elapsed = (
        time.time()
        - file_start
    )

    print()
    print("========================================")
    print("RUN COMPLETE")
    print("========================================")

    print(
        f"Processed this run : "
        f"{len(pending):,}"
    )

    print(
        f"Elapsed time       : "
        f"{total_elapsed:.2f}s"
    )

    print(
        f"Output             : "
        f"{output_file}"
    )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run OpenAI, Anthropic, or Gemini generation "
            "on one 10K benchmark request JSONL."
        )
    )

    parser.add_argument(
        "--request-file",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-file",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--anthropic-api-key",
        default=None,
        help="Anthropic API key, required for Anthropic request files",
    )

    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="OpenAI API key, required for OpenAI request files",
    )

    parser.add_argument(
        "--gemini-api-key",
        default=None,
        help="Gemini API key, required for Gemini request files",
    )

    args = parser.parse_args()

    request_file = (
        args.request_file
    )

    output_file = (
        args.output_file
    )

    if not request_file.exists():

        raise FileNotFoundError(
            f"Request file not found: "
            f"{request_file}"
        )

    ensure_dir(
        output_file.parent
    )

    requests = load_jsonl(
        request_file
    )

    if not requests:

        raise RuntimeError(
            f"No requests found in: "
            f"{request_file}"
        )

    # All requests in one request file should use
    # exactly one provider.
    providers = {
        row.get("provider")
        for row in requests
    }

    if len(providers) != 1:

        raise RuntimeError(
            "Request file contains multiple providers: "
            f"{providers}"
        )

    provider = next(
        iter(providers)
    )

    print()
    print("========================================")
    print("PROVIDER")
    print("========================================")

    print(
        f"Detected provider: "
        f"{provider}"
    )

    clients: Dict[
        str,
        Any,
    ] = {}

    if provider == "openai":

        clients["openai"] = (
            get_openai_client(
                args.openai_api_key
            )
        )

    elif provider == "anthropic":

        clients["anthropic"] = (
            get_anthropic_client(
                args.anthropic_api_key
            )
        )

    elif provider == "gemini":

        clients["gemini"] = (
            get_gemini_client(
                args.gemini_api_key
            )
        )

    else:

        raise ValueError(
            f"Unsupported provider: "
            f"{provider}"
        )

    process_request_file(
        request_file=request_file,
        output_file=output_file,
        clients=clients,
    )


if __name__ == "__main__":
    main()