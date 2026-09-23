#!/usr/bin/env python3

"""
response_generator.py

Generate responses from ONE already-filtered multilingual AEGIS JSONL file.

Required CLI arguments:
    --input-file
    --model-name
    --model-family
    --output-file

Example:

    python response_generator.py \
        --input-file filtered_aegis_multilang.jsonl \
        --model-name CohereForAI/aya-expanse-32b \
        --model-family aya \
        --output-file aya_expanse_32b.jsonl

Input identity:
    (root_id, language, category)

Output:
    ONE JSONL file directly.

There is:
    - no CSV filtering
    - no batch directory
    - no intermediate shard files
    - no merge logic
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ================================================================
# GENERATION CONFIGURATION
# ================================================================

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
SEED = 42

# Change if necessary according to GPU/model size.
BATCH_SIZE = 4


# ================================================================
# JSON-SAFE VALUE
# ================================================================

def clean_json_value(value):
    """
    Convert NaN / Inf to None so output remains valid JSON.

    Example:
        COMET NaN -> null
    """

    if value is None:
        return None

    if isinstance(value, float):

        if math.isnan(value) or math.isinf(value):
            return None

    return value


# ================================================================
# LOAD FILTERED AEGIS DATA
# ================================================================

def load_filtered_data(input_file):
    """
    Read the already-filtered AEGIS JSONL.

    Required input fields:
        root_id
        language
        category
        prompt_translated

    Duplicate identity:
        (root_id, language, category)

    Duplicate triples are skipped.
    """

    records = []

    seen_triples = set()

    invalid_json = 0
    missing_required = 0
    duplicate_triples = 0


    required_fields = [
        "root_id",
        "language",
        "category",
        "prompt_translated",
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


            # ----------------------------------------------------
            # Parse JSON
            # ----------------------------------------------------

            try:

                obj = json.loads(line)

            except json.JSONDecodeError:

                invalid_json += 1

                print(
                    f"[WARN] Invalid JSON at line "
                    f"{line_num}"
                )

                continue


            # ----------------------------------------------------
            # Verify required fields
            # ----------------------------------------------------

            missing = [
                field
                for field in required_fields
                if obj.get(field) is None
            ]


            if missing:

                missing_required += 1

                print(
                    f"[WARN] Line {line_num}: "
                    f"missing {missing}"
                )

                continue


            # ----------------------------------------------------
            # Clean identity fields
            # ----------------------------------------------------

            root_id = str(
                obj["root_id"]
            ).strip()

            language = str(
                obj["language"]
            ).strip()

            category = str(
                obj["category"]
            ).strip()

            prompt = str(
                obj["prompt_translated"]
            ).strip()


            if not prompt:

                missing_required += 1

                print(
                    f"[WARN] Empty prompt at "
                    f"line {line_num}"
                )

                continue


            # ----------------------------------------------------
            # Triple identity
            # ----------------------------------------------------

            triple = (
                root_id,
                language,
                category,
            )


            if triple in seen_triples:

                duplicate_triples += 1

                print(
                    f"[WARN] Duplicate triple skipped: "
                    f"{triple}"
                )

                continue


            seen_triples.add(
                triple
            )


            # Store cleaned values.
            obj["root_id"] = root_id
            obj["language"] = language
            obj["category"] = category
            obj["prompt_translated"] = prompt


            records.append(
                obj
            )


    print("\n========================================")
    print("INPUT SUMMARY")
    print("========================================")

    print(
        f"Valid unique records      : "
        f"{len(records):,}"
    )

    print(
        f"Duplicate triples skipped : "
        f"{duplicate_triples:,}"
    )

    print(
        f"Invalid JSON rows         : "
        f"{invalid_json:,}"
    )

    print(
        f"Missing required fields   : "
        f"{missing_required:,}"
    )


    return records


# ================================================================
# RESUME SUPPORT
# ================================================================

def load_processed_keys(
    output_file,
    model_name,
):
    """
    Read existing output file and find already-completed samples.

    Resume identity:

        (
            root_id,
            language,
            category,
            model
        )

    This means rerunning the same command will not regenerate
    responses already present in the output file.
    """

    processed = set()


    if not output_file.exists():

        print(
            "\nNo existing output file. "
            "Starting new generation."
        )

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

            except Exception:
                continue


            root_id = obj.get(
                "root_id"
            )

            language = obj.get(
                "language"
            )

            category = (
                obj.get(
                    "eval",
                    {}
                ).get(
                    "category"
                )
            )

            stored_model = (
                obj.get(
                    "meta",
                    {}
                ).get(
                    "model"
                )
            )


            if (
                root_id is None
                or language is None
                or category is None
                or stored_model is None
            ):

                continue


            key = (
                str(root_id).strip(),
                str(language).strip(),
                str(category).strip(),
                str(stored_model).strip(),
            )


            processed.add(
                key
            )


    print(
        f"Already processed         : "
        f"{len(processed):,}"
    )


    return processed


# ================================================================
# JSONL WRITER
# ================================================================

def append_jsonl(
    output_file,
    records,
):
    """
    Append generated responses directly to final output JSONL.
    """

    with output_file.open(
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


# ================================================================
# MODEL
# ================================================================

class ResponseGenerator:

    def __init__(
        self,
        model_name,
        model_family,
    ):

        self.model_name = model_name
        self.model_family = model_family


        # --------------------------------------------------------
        # Reproducibility
        # --------------------------------------------------------

        torch.manual_seed(
            SEED
        )


        if torch.cuda.is_available():

            torch.cuda.manual_seed_all(
                SEED
            )


        print("\n========================================")
        print("LOADING MODEL")
        print("========================================")

        print(
            f"Model name   : "
            f"{self.model_name}"
        )

        print(
            f"Model family : "
            f"{self.model_family}"
        )


        # --------------------------------------------------------
        # Tokenizer
        # --------------------------------------------------------

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True,
            )
        )


        if self.tokenizer.pad_token is None:

            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )


        self.tokenizer.padding_side = (
            "left"
        )


        # --------------------------------------------------------
        # Model
        # --------------------------------------------------------

        self.model = (
            AutoModelForCausalLM
            .from_pretrained(
                self.model_name,

                torch_dtype=(
                    torch.bfloat16
                    if torch.cuda.is_available()
                    else torch.float32
                ),

                device_map="auto",

                trust_remote_code=True,

                attn_implementation="sdpa",
            )
        )


        self.model.eval()


        print(
            "\nModel loaded successfully."
        )


    # ============================================================
    # FORMAT CHAT PROMPTS
    # ============================================================

    def format_prompts(
        self,
        prompts,
    ):

        formatted_prompts = []


        for prompt in prompts:

            messages = [

                {
                    "role": "system",
                    "content":
                        "You are a helpful, safe, "
                        "and honest assistant.",
                },

                {
                    "role": "user",
                    "content": prompt,
                },
            ]


            try:

                formatted = (
                    self.tokenizer
                    .apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )


            except Exception:

                formatted = (
                    f"User: {prompt}\n"
                    f"Assistant:"
                )


            formatted_prompts.append(
                formatted
            )


        return formatted_prompts


    # ============================================================
    # COUNT COMPLETION TOKENS
    # ============================================================

    def count_generated_tokens(
        self,
        token_ids,
    ):
        """
        Count actual generated tokens.

        Stops at EOS when present.
        """

        ids = token_ids.tolist()

        eos_id = (
            self.tokenizer.eos_token_id
        )


        if (
            eos_id is not None
            and eos_id in ids
        ):

            return (
                ids.index(eos_id)
                + 1
            )


        pad_id = (
            self.tokenizer.pad_token_id
        )


        if pad_id is not None:

            while (
                ids
                and ids[-1] == pad_id
            ):

                ids.pop()


        return len(ids)


    # ============================================================
    # GENERATE BATCH
    # ============================================================

    def generate_batch(
        self,
        prompts,
        languages,
    ):

        formatted_prompts = (
            self.format_prompts(
                prompts
            )
        )


        # --------------------------------------------------------
        # Maximum model context
        # --------------------------------------------------------

        max_input_length = getattr(
            self.model.config,
            "max_position_embeddings",
            4096,
        )


        # --------------------------------------------------------
        # Tokenization
        # --------------------------------------------------------

        inputs = self.tokenizer(
            formatted_prompts,

            return_tensors="pt",

            padding=True,

            truncation=True,

            max_length=max_input_length,

        ).to(
            self.model.device
        )


        input_lengths = (
            inputs[
                "attention_mask"
            ].sum(
                dim=1
            )
        )


        # Number of tokens occupied by padded batch input.
        padded_input_width = (
            inputs[
                "input_ids"
            ].shape[1]
        )


        # --------------------------------------------------------
        # Generation
        # --------------------------------------------------------

        generation_start = (
            time.time()
        )


        with torch.no_grad():

            outputs = (
                self.model.generate(
                    **inputs,

                    max_new_tokens=
                        MAX_NEW_TOKENS,

                    do_sample=True,

                    temperature=
                        TEMPERATURE,

                    top_p=
                        TOP_P,

                    pad_token_id=
                        self.tokenizer
                        .pad_token_id,

                    eos_token_id=
                        self.tokenizer
                        .eos_token_id,
                )
            )


        generation_time = (
            time.time()
            - generation_start
        )


        responses = []

        metadata = []


        # --------------------------------------------------------
        # Decode continuation only
        # --------------------------------------------------------

        for i in range(
            len(prompts)
        ):

            generated_ids = (
                outputs[i][
                    padded_input_width:
                ]
            )


            response = (
                self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
                .strip()
            )


            prompt_tokens = int(
                input_lengths[i]
                .item()
            )


            completion_tokens = (
                self.count_generated_tokens(
                    generated_ids
                )
            )


            meta = {

                "model":
                    self.model_name,

                "model_family":
                    self.model_family,

                "temperature":
                    TEMPERATURE,

                "top_p":
                    TOP_P,

                "max_new_tokens":
                    MAX_NEW_TOKENS,

                "language":
                    languages[i],

                "prompt_tokens":
                    prompt_tokens,

                "completion_tokens":
                    completion_tokens,

                "total_tokens":
                    (
                        prompt_tokens
                        + completion_tokens
                    ),

                "prompt_length_chars":
                    len(
                        prompts[i]
                    ),

                "response_length_chars":
                    len(
                        response
                    ),

                "time_per_sample_sec":
                    (
                        generation_time
                        / len(prompts)
                    ),
            }


            responses.append(
                response
            )


            metadata.append(
                meta
            )


        return (
            responses,
            metadata,
        )


# ================================================================
# BUILD OUTPUT RECORD
# ================================================================

def build_response_record(
    item,
    response,
    meta,
    input_file,
):
    """
    Convert one filtered AEGIS record into response format.
    """

    return {

        # --------------------------------------------------------
        # Identity
        # --------------------------------------------------------

        "root_id":
            item["root_id"],

        "language":
            item["language"],


        # --------------------------------------------------------
        # Prompt and generated response
        # --------------------------------------------------------

        "prompt":
            item[
                "prompt_translated"
            ],

        "response":
            response,


        # --------------------------------------------------------
        # Generation metadata
        # --------------------------------------------------------

        "meta":
            meta,


        # --------------------------------------------------------
        # Evaluation metadata
        # --------------------------------------------------------

        "eval": {

            "category":
                item.get(
                    "category"
                ),

            "tier":
                item.get(
                    "tier"
                ),

            "label":
                item.get(
                    "label"
                ),

            "f1":
                clean_json_value(
                    item.get(
                        "f1"
                    )
                ),

            "comet":
                clean_json_value(
                    item.get(
                        "comet"
                    )
                ),

            "combined_score":
                clean_json_value(
                    item.get(
                        "combined_score"
                    )
                ),

            "quality_bucket":
                item.get(
                    "quality_bucket"
                ),
        },


        # --------------------------------------------------------
        # Source dataset
        # --------------------------------------------------------

        "source_file":
            input_file.name,
    }


# ================================================================
# RUN GENERATION
# ================================================================

def run_generation(
    input_file,
    model_name,
    model_family,
    output_file,
):

    input_file = Path(
        input_file
    )

    output_file = Path(
        output_file
    )


    # ------------------------------------------------------------
    # Verify input
    # ------------------------------------------------------------

    if not input_file.exists():

        raise FileNotFoundError(
            f"Input file not found: "
            f"{input_file}"
        )


    # ------------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    print("\n========================================")
    print("RESPONSE GENERATION")
    print("========================================")

    print(
        f"Input file   : "
        f"{input_file}"
    )

    print(
        f"Output file  : "
        f"{output_file}"
    )

    print(
        f"Model        : "
        f"{model_name}"
    )

    print(
        f"Model family : "
        f"{model_family}"
    )

    print(
        f"Batch size   : "
        f"{BATCH_SIZE}"
    )


    # ------------------------------------------------------------
    # Load input dataset
    # ------------------------------------------------------------

    data = load_filtered_data(
        input_file
    )


    # ------------------------------------------------------------
    # Existing outputs / resume
    # ------------------------------------------------------------

    processed = load_processed_keys(
        output_file,
        model_name,
    )


    to_process = []

    skipped = 0


    for item in data:

        key = (

            item["root_id"],

            item["language"],

            item["category"],

            model_name,
        )


        if key in processed:

            skipped += 1

        else:

            to_process.append(
                item
            )


    print("\n========================================")
    print("PROCESSING PLAN")
    print("========================================")

    print(
        f"Input records      : "
        f"{len(data):,}"
    )

    print(
        f"Already completed  : "
        f"{skipped:,}"
    )

    print(
        f"To generate        : "
        f"{len(to_process):,}"
    )


    if not to_process:

        print(
            "\nAll responses already exist."
        )

        return


    # ------------------------------------------------------------
    # Load model once
    # ------------------------------------------------------------

    generator = ResponseGenerator(
        model_name=model_name,
        model_family=model_family,
    )


    generated_total = 0


    # ============================================================
    # Batch generation
    # ============================================================

    for start in range(
        0,
        len(to_process),
        BATCH_SIZE,
    ):

        batch = (
            to_process[
                start:
                start + BATCH_SIZE
            ]
        )


        prompts = [
            item[
                "prompt_translated"
            ]
            for item in batch
        ]


        languages = [
            item[
                "language"
            ]
            for item in batch
        ]


        responses, metas = (
            generator.generate_batch(
                prompts=prompts,
                languages=languages,
            )
        )


        output_records = []


        for i, item in enumerate(
            batch
        ):

            output_record = (
                build_response_record(
                    item=item,
                    response=responses[i],
                    meta=metas[i],
                    input_file=input_file,
                )
            )


            output_records.append(
                output_record
            )


        # --------------------------------------------------------
        # Save immediately.
        #
        # This enables safe resume if process stops.
        # --------------------------------------------------------

        append_jsonl(
            output_file,
            output_records,
        )


        generated_total += (
            len(output_records)
        )


        print(
            f"Generated "
            f"{generated_total:,}/"
            f"{len(to_process):,}"
            f" | Overall "
            f"{skipped + generated_total:,}/"
            f"{len(data):,}"
        )


    print("\n========================================")
    print("GENERATION COMPLETE")
    print("========================================")

    print(
        f"New responses generated : "
        f"{generated_total:,}"
    )

    print(
        f"Output file             : "
        f"{output_file}"
    )


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Generate LLM responses from one "
            "filtered multilingual AEGIS JSONL."
        )
    )


    parser.add_argument(
        "--input-file",
        required=True,
        type=Path,
        help=(
            "Filtered AEGIS multilingual "
            "JSONL file"
        ),
    )


    parser.add_argument(
        "--model-name",
        required=True,
        type=str,
        help=(
            "Hugging Face model name, e.g. "
            "CohereForAI/aya-expanse-32b"
        ),
    )


    parser.add_argument(
        "--model-family",
        required=True,
        type=str,
        help=(
            "Model family, e.g. "
            "aya, llama, qwen, mistral"
        ),
    )


    parser.add_argument(
        "--output-file",
        required=True,
        type=Path,
        help=(
            "Final response JSONL file"
        ),
    )


    args = parser.parse_args()


    run_generation(
        input_file=
            args.input_file,

        model_name=
            args.model_name,

        model_family=
            args.model_family,

        output_file=
            args.output_file,
    )