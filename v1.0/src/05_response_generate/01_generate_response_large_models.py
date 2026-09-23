# =========================================================
# 🔷 IMPORTS
# =========================================================
import json
import os
import time
from pathlib import Path
import argparse
import pandas as pd
from openai import OpenAI

# =========================================================
# 🔷 MINDROUTER API CALL
# =========================================================
def call_mindrouter_api(
    prompt,
    model="openai/gpt-oss-120b",
    api_key="YOUR_API_KEY",
    base_url="https://mindrouter.uidaho.edu/v1",
    temperature=0,
    top_p=1,
    max_tokens=512
):
    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful, safe, and honest assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        
        choice = response.choices[0]

        content = choice.message.content

        if content is None:
            content = getattr(choice.message, "reasoning_content", None)
        #print(content)
        return {
            "response": content,
            "error": None
        }

    except Exception as e:
        return {
            "response": "ERROR",
            "error": str(e)
        }


def mind_router(params):
    prompt = params.get("prompt")

    result = call_mindrouter_api(
        prompt=prompt,
        model=params.get("model", "openai/gpt-oss-120b"),
        api_key=params.get("api_key", "YOUR_API_KEY"),
        temperature=params.get("temperature", 0),
        top_p=params.get("top_p", 1),
        max_tokens=params.get("max_tokens", 512)
    )

    # call_mindrouter_api returns (None, error_str) on failure
    if isinstance(result, tuple):
        return {
            "response":               "ERROR",
            "error":                  result[1],
            "prompt_length_tokens":   0,
            "response_length_tokens": 0,
            "total_tokens":           0,
        }

    return {
        "response":               result.choices[0].message.content,
        "error":                  None,
        # ── Pull real token counts from API usage object ──────────
        "prompt_length_tokens":   result.usage.prompt_tokens,
        "response_length_tokens": result.usage.completion_tokens,
        "total_tokens":           result.usage.total_tokens,
    }


# =========================================================
# 🔷 LOAD FILTER + META
# =========================================================
def load_filter_data(filter_file):
    df_filter = pd.read_csv(filter_file)

    filter_map = {}
    filter_meta = {}

    for _, row in df_filter.iterrows():
        rid = str(row["root_id"])
        lang = str(row["language"])

        if rid not in filter_map:
            filter_map[rid] = set()
        filter_map[rid].add(lang)

        filter_meta[(rid, lang)] = {
            "category": row.get("category"),
            "tier": row.get("tier"),
            "label": row.get("label"),
            "f1": row.get("f1"),
            "comet": row.get("comet"),
            "combined_score": row.get("combined_score"),
            "quality_bucket": row.get("quality_bucket"),
        }

    print(f"✅ Loaded filter metadata for {len(filter_map)} root_ids")
    return filter_map, filter_meta


# =========================================================
# 🔷 FLATTEN WITH FILTER
# =========================================================
def flatten_prompts_filtered(data, filter_map):
    flat_data = []

    for obj in data:
        rid = obj["id"]

        if rid not in filter_map:
            continue

        for lang, trans in obj.get("translation", {}).items():
            if lang not in filter_map[rid]:
                continue

            prompt = trans.get("prompt_translated_lang")

            if prompt:
                flat_data.append({
                    "root_id": rid,
                    "language": lang,
                    "prompt": prompt
                })

    print(f"🚀 Filtered samples: {len(flat_data)}")
    return flat_data


# =========================================================
# 🔷 RESUME SUPPORT
# =========================================================
def load_processed_keys(file_path):
    processed = set()

    if not os.path.exists(file_path):
        return processed

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                key = (
                    obj["root_id"],
                    obj["language"],
                    obj["meta"]["model"]
                )
                processed.add(key)
            except:
                continue

    print(f"🔁 Loaded processed: {len(processed)}")
    return processed


# =========================================================
# 🔷 SAVE JSONL
# =========================================================
def append_to_jsonl(file_path, records):
    with open(file_path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =========================================================
# 🔷 GENERATOR (API BASED)
# =========================================================
class UnifiedResponseGenerator:
    def __init__(self, model_name, max_new_tokens=512, temperature=0, top_p=1, seed=42, api_key="YOUR_API_KEY"):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.api_key = api_key

        print(f"🚀 Using API model: {model_name}")

    def generate_batch(self, prompts, languages):
        responses = []
        metadata = []

        batch_start = time.time()

        for i, prompt in enumerate(prompts):
            start = time.time()

            result = mind_router({
                "prompt": prompt,
                "model": self.model_name,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_new_tokens,
                "api_key": self.api_key
            })

            response_text = result["response"]

            end = time.time()

            meta = {
                # ── Model config ──────────────────────────────────
                "model":          self.model_name,
                "temperature":    self.temperature,
                "top_p":          self.top_p,
                "max_new_tokens": self.max_new_tokens,
                "seed": self.seed,
                
                # ── Language ──────────────────────────────────────
                "language":       languages[i],
                
                # ── Token counts (from API usage object) ──────────
                "prompt_length_tokens":   result["prompt_length_tokens"],
                "response_length_tokens": result["response_length_tokens"],
                "total_tokens":           result["total_tokens"],
                
                # ── Character counts ──────────────────────────────
                "prompt_length_chars":   len(prompt),
                "response_length_chars": len(response_text) if response_text != "ERROR" else 0,
                # ── Timing ────────────────────────────────────────
                "time_per_sample_sec": end - start,
                "was_truncated": 0
            }

            if result["error"]:
                meta["error"] = result["error"]

            responses.append(response_text)
            metadata.append(meta)

        total_time = time.time() - batch_start

        for m in metadata:
            m["batch_samples_per_sec"] = len(prompts) / total_time

        return responses, metadata


# =========================================================
# 🔷 PIPELINE
# =========================================================
def generate_and_save(generator, flat_data, output_file, filter_meta):
    processed = load_processed_keys(output_file)
    buffer = []

    for i in range(0, len(flat_data), 32):
        batch = flat_data[i:i+32]

        prompts = [x["prompt"] for x in batch]
        langs = [x["language"] for x in batch]

        responses, metas = generator.generate_batch(prompts, langs)

        for j, item in enumerate(batch):
            key = (item["root_id"], item["language"], generator.model_name)

            if key in processed:
                continue

            record = {
                "root_id": item["root_id"],
                "language": item["language"],
                "prompt": item["prompt"],
                "response": responses[j],
                "meta": metas[j],
                "eval": filter_meta.get((item["root_id"], item["language"]), {})
            }

            buffer.append(record)

        if len(buffer) >= 100:
            append_to_jsonl(output_file, buffer)
            print(f"💾 Saved {len(buffer)}")
            buffer = []

    if buffer:
        append_to_jsonl(output_file, buffer)

    print("✅ DONE")


# =========================================================
# 🔷 MAIN
# =========================================================
def main(model_name, model_family, data_dir, filter_file, output_base_dir, api_key):

    print(f"\n🚀 Running model: {model_name}")

    model_tag = model_name.split("/")[-1].lower().replace("-", "_")

    filter_map, filter_meta = load_filter_data(filter_file)

    for batch_num in range(12, 59):
        input_file = f"{data_dir}/batch_{batch_num}.json"

        if not os.path.exists(input_file):
            continue

        print(f"\n📦 Processing batch {batch_num}")

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        flat_data = flatten_prompts_filtered(data, filter_map)

        out_dir = f"{output_base_dir}/{model_family}/{model_tag}"
        os.makedirs(out_dir, exist_ok=True)

        output_file = f"{out_dir}/{model_tag}_{batch_num}.jsonl"

        generator = UnifiedResponseGenerator(model_name, api_key=api_key)

        generate_and_save(generator, flat_data, output_file, filter_meta)

    print("\n✅ ALL BATCHES COMPLETED")


# =========================================================
# 🔷 CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-family", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--filter-file", type=str, required=True)
    parser.add_argument("--output-base-dir", type=str, required=True)
    parser.add_argument("--api-key", type=str, required=True)

    args = parser.parse_args()

    main(
        model_name=args.model_name,
        model_family=args.model_family,
        data_dir=args.data_dir,
        filter_file=args.filter_file,
        output_base_dir=args.output_base_dir,
        api_key=args.api_key
    )