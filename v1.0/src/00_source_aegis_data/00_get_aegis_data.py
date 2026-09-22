#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert the Aegis AI Content Safety (train split) prompts to Bangla using a
translation model. Only rows with BOTH prompt_label and response_label are kept.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any

from datasets import load_dataset  # type: ignore
from transformers import pipeline  # type: ignore
from transformers import AutoTokenizer, pipeline


def translate_batch(texts: List[str], translator) -> List[str]:
    outputs = translator(texts, max_length=512)
    return [o["translation_text"] for o in outputs]


def parse_categories(violated: Optional[str]) -> (str, Optional[str]):
    if not violated:
        return "Safe", None
    parts = [p.strip() for p in violated.split(",") if p.strip()]
    if not parts:
        return "Safe", None
    primary = parts[0]
    secondary = parts[1] if len(parts) > 1 else None
    return primary, secondary


# -- Split the violated_categories string and get all unique values
# SELECT DISTINCT trim(unnested_category) AS category
# FROM (
#     SELECT unnest(string_split(violated_categories, ',')) AS unnested_category
#     FROM train
#     WHERE violated_categories IS NOT NULL
# )
# WHERE unnested_category != ''
# ORDER BY category ASC;

import json
from datasets import load_dataset

# ==============================
# Severity Ranking
# ==============================
severity_order = [
    "Sexual (minor)",
    "Suicide and Self Harm",
    "Threat",
    "Violence",
    "Guns and Illegal Weapons",
    "Criminal Planning/Confessions",
    "Illegal Activity",
    "Malware",
    "Fraud/Deception",
    "Controlled/Regulated Substances",
    "Hate/Identity Hate",
    "Harassment",
    "Sexual",
    "Political/Misinformation/Conspiracy",
    "High Risk Gov Decision Making",
    "Unauthorized Advice",
    "PII/Privacy",
    "Manipulation",
    "Immoral/Unethical",
    "Profanity",
    "Copyright/Trademark/Plagiarism",
    "Needs Caution",
    "Other"
]

# strip to be safe
severity_map = {cat.strip(): i + 1 for i, cat in enumerate(severity_order)}

def select_most_severe(categories):

    if isinstance(categories, str):
        categories = [categories]   # convert to list
    elif isinstance(categories, (list, tuple, set)):
        categories = list(categories)
    else:
        categories = []

    if not categories:
        return None
    
    valid = [c for c in categories if c in severity_map]
    

    if not valid:
        return None
    return min(valid, key=lambda c: severity_map[c])


def main() -> None:
    print("Loading Aegis training split…")
    dataset = load_dataset(
        "nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train"
    )

    import os
    model_name = os.getenv("AegisTranslatorModel", "shhossain/opus-mt-en-to-bn")
    print(f"Initialising translation model: {model_name}")
    # translator = pipeline("translation", model=model_name, device=0)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("tokenizer.model_max_length: ", tokenizer.model_max_length)
    

    translator = pipeline(
        "translation",
        model=model_name,
        src_lang="eng_Latn",
        tgt_lang="ben_Beng",
        device=0,
        max_length=1024,   # output max tokens
        truncation=True
    )

    records: List[Dict[str, Any]] = [ex for ex in dataset]
    converted: List[Dict[str, Optional[str]]] = []
    total = len(records)
    print(f"Processing {total} prompts…")

    records = []


    for idx, row in enumerate(dataset):

        # print("prompt token length: ", len(tokenizer(row.get("prompt"))["input_ids"]))
        # print("response token length: ", len(tokenizer(row.get("response"))["input_ids"]))

        categories = row.get("violated_categories", [])

        most_severe = select_most_severe(categories)

        prompt_text = row.get("prompt")

        if not prompt_text or not isinstance(prompt_text, str) and len(tokenizer(prompt_text)["input_ids"]) > 1024:
            prompt_bn = None
        else:
            try:
                prompt_bn = translator(prompt_text)[0]["translation_text"]
            except Exception as e:
                print(f"  ⚠️ Error translating record {idx + 1}: {e}. Using truncated English text.")
                prompt_bn = prompt_text  # fallback

            # -----------------------------
        # Translate response safely
        # -----------------------------
        response_text = row.get("response")

        if not response_text or not isinstance(response_text, str) and len(tokenizer(response_text)["input_ids"]) > 1024:
            response_bn = None
        else:
            try:
                response_bn = translator(response_text)[0]["translation_text"]
            except Exception as e:
                print(f"  ⚠️ Error translating record {idx + 1}: {e}. Using truncated English text.")
                response_bn = response_text  # fallback

        # -------------------------
        # Handle prompt_label_source
        # -------------------------
        prompt_label_source = row.get("prompt_label_source")
        if not prompt_label_source:
            prompt_label_source = None

        # -------------------------
        # Handle response_label_source
        # -------------------------
        response_label_source = row.get("response_label_source")
        if not response_label_source:
            response_label_source = None

        # --------------------------------------
        # Apply filters BEFORE building the record
        # --------------------------------------

        if (prompt_label_source is None 
                or response_label_source is None 
                or response_bn is None or prompt_bn is None or most_severe is None):
            continue   # skip this row entirely

        print(f"row at idx: {idx} passed all filters.")
        record = {
            "id": row.get("id"),
            "reconstruction_id_if_redacted": row.get("reconstruction_id_if_redacted"),
            "prompt": row.get("prompt"),
            "response": row.get("response"),
            "prompt_bn": prompt_bn,
            "response_bn": response_bn,
            "prompt_label": row.get("prompt_label"),
            "response_label": row.get("response_label"),

            # original Aegis list
            "violated_categories": categories,

            # NEW: most severe category
            "most_severe_category": most_severe,

            "prompt_label_source": prompt_label_source,
            "response_label_source": response_label_source
        }

        records.append(record)
        print("records.len: ", len(records))

    # ==============================
    # Save Output
    # ==============================
    output_file = "/home/malam10/projects/ai-safety-bangla/final/data/converted_aegis_v2_en_bn.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(records)} records to {output_file}")


if __name__ == "__main__":
    main()
