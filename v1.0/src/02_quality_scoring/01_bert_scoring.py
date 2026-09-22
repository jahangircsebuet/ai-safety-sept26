import os
import json
from tqdm import tqdm
from bert_score import score as bertscore
import numpy as np
import pandas as pd

DATA_DIR = "/home/malam/projects/benchmarks/ai-safety-sept26/v1.0/data/aegis_batches"
MODEL_TYPE = "microsoft/deberta-xlarge-mnli"
BATCH_SIZE = 32
# language tier map
HIGH = set([
        "Arabic",
        "Chinese (Simplified)",
        "Chinese (Traditional)",
        "English",
        "French",
        "German",
        "Italian",
        "Japanese",
        "Korean",
        "Portuguese",
        "Russian",
        "Spanish"
    ])

MEDIUM = set([
        "Bengali",
        "Bulgarian",
        "Czech",
        "Danish",
        "Dutch",
        "Finnish",
        "Greek",
        "Hebrew",
        "Hindi",
        "Indonesian",
        "Malay",
        "Norwegian",
        "Persian",
        "Polish",
        "Romanian",
        "Swedish",
        "Thai",
        "Turkish",
        "Ukrainian",
        "Urdu",
        "Vietnamese"
    ])

def get_tier(lang):
    if lang in HIGH:
        return "high"
    elif lang in MEDIUM:
        return "medium"
    else:
        return "low"

# -------- FUNCTION --------
def compute_bertscore_batch(src_texts, back_texts):
    P, R, F1 = bertscore(
        cands=back_texts,
        refs=src_texts,
        model_type=MODEL_TYPE,
        batch_size=BATCH_SIZE,
        rescale_with_baseline=True,
        lang="en"
    )
    return F1.tolist()

# -------- STORAGE --------
rows = []                  # MAIN TABLE
language_scores = {}       # summary

# -------- MAIN LOOP --------
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".json")])

for file_name in files:
    file_path = os.path.join(DATA_DIR, file_name)
    print(f"\nProcessing file: {file_name}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Temporary storage per language
    lang_src = {}
    lang_back = {}
    lang_meta = {}

    for idx, item in enumerate(data):
        src = item.get("prompt", "")
        category = item.get("violated_categories", "unknown")
        label = item.get("prompt_label", "unknown")
        root_id = item.get("id", f"{file_name}_{idx}")

        translations = item.get("translation", {})

        for lang, trans in translations.items():
            back = trans.get("prompt_back_to_original_lang", None)

            if not back:
                continue

            if lang not in lang_src:
                lang_src[lang] = []
                lang_back[lang] = []
                lang_meta[lang] = []

            lang_src[lang].append(src)
            lang_back[lang].append(back)

            lang_meta[lang].append({
                "root_id": root_id,
                "category": category,
                "label": label,
                "prompt_len": len(src),
                "tier": get_tier(lang)
            })

    # -------- Compute BERTScore --------
    for lang in tqdm(lang_src.keys(), desc=f"{file_name} languages"):
        scores = compute_bertscore_batch(lang_src[lang], lang_back[lang])

        if lang not in language_scores:
            language_scores[lang] = []

        language_scores[lang].extend(scores)

        # -------- Build row-level table --------
        for i, f1 in enumerate(scores):
            meta = lang_meta[lang][i]

            rows.append({
                "root_id": meta["root_id"],
                "language": lang,
                "category": meta["category"],
                "tier": meta["tier"],
                "f1": f1,
                "prompt_len": meta["prompt_len"],
                "label": meta["label"],
                "failure": int(f1 < 0.8),
                "severe_failure": int(f1 < 0.7)
            })

# -------- SAVE MAIN TABLE --------
df = pd.DataFrame(rows)
df.to_csv("/home/malam/projects/benchmarks/ai-safety-sept26/v1.0/data/02_quality_scores/bert_scores.csv", index=False)

print("\nSaved evaluation_table.csv")

# -------- AGGREGATE LANGUAGE SUMMARY --------
summary = {}

for lang, scores in language_scores.items():
    summary[lang] = {
        "mean_f1": float(np.mean(scores)),
        "std_f1": float(np.std(scores)),
        "count": len(scores)
    }

with open("/home/malam/projects/benchmarks/ai-safety-sept26/v1.0/data/02_quality_scores/bert_language_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Saved language_summary.json")