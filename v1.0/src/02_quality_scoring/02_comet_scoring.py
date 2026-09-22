import os
import json
import time
import pandas as pd
from tqdm import tqdm
from comet import download_model, load_from_checkpoint
import numpy as np

# ===============================
# CONFIG
# ===============================
DATA_DIR = "/home/malam10/projects/ai-safety-bangla/latest/dataset/aegis_batches"
OUTPUT_DIR = "/home/malam/projects/benchmarks/ai-safety-sept26/v1.0/data/02_quality_scores"
LOG_FILE = "/home/malam/projects/benchmarks/ai-safety-sept26/v1.0/logs/comet_processing.log"
SUMMARY_FILE = "/home/malam/projects/benchmarks/ai-safety-sept26/v1.0/data/02_quality_scores/comet_language_summary.json"
EVAL_PATH = "/home/malam/projects/benchmarks/ai-safety-sept26/v1.0/data/02_quality_scores/comet_scores.csv"

BATCH_SIZE = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# SUPPORTED LANGS
# ===============================
SUPPORTED_LANGS = set([
    "Afrikaans","Amharic","Arabic","Armenian","Assamese","Azerbaijani",
    "Belarusian","Bengali","Bosnian","Bulgarian","Burmese","Catalan",
    "Chinese (Simplified)","Chinese (Traditional)","Croatian","Czech",
    "Danish","Dutch","English","Estonian","Filipino (Tagalog)","Finnish",
    "French","Galician","Georgian","German","Greek","Gujarati","Hausa",
    "Hebrew","Hindi","Hungarian","Icelandic","Indonesian","Irish","Italian",
    "Japanese","Javanese","Kannada","Kazakh","Khmer","Korean","Kyrgyz",
    "Lao","Latvian","Lithuanian","Macedonian","Malay","Malayalam","Marathi",
    "Mongolian","Nepali","Norwegian","Oriya","Oromo","Pashto","Persian",
    "Polish","Portuguese","Punjabi","Romanian","Russian","Serbian","Sindhi",
    "Sinhala","Slovak","Slovenian","Somali","Spanish","Swahili","Swedish",
    "Tamil","Telugu","Thai","Turkish","Ukrainian","Urdu","Uzbek","Vietnamese",
    "Welsh","Xhosa"
])

# ===============================
# LOAD MODEL
# ===============================
model_path = download_model("Unbabel/wmt22-cometkiwi-da")
model = load_from_checkpoint(model_path)

# ===============================
# LOGGING FUNCTION
# ===============================
def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

# ===============================
# MAIN LOOP
# ===============================
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".json")])

total_processed = 0
start_global = time.time()

all_rows = []

for file_name in files:
    file_path = os.path.join(DATA_DIR, file_name)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    log(f"\nProcessing file: {file_name} | total items: {len(data)}")

    # ===============================
    # BATCHING
    # ===============================
    for batch_idx in range(0, len(data), BATCH_SIZE):
        batch = data[batch_idx: batch_idx + BATCH_SIZE]

        batch_start = time.time()

        batch_data = []
        meta_info = []
        rows = []

        # ===============================
        # PREPARE DATA
        # ===============================
        for idx, item in enumerate(batch):
            src = item.get("prompt", "")
            category = item.get("violated_categories", "unknown")
            label = item.get("prompt_label", "unknown")
            root_id = item.get("id", f"{file_name}_{batch_idx+idx}")

            translations = item.get("translation", {})

            for lang, trans in translations.items():
                mt = trans.get("prompt_translated_lang", None)

                if not mt:
                    continue

                # Unsupported → None
                if lang not in SUPPORTED_LANGS:
                    rows.append({
                        "root_id": root_id,
                        "language": lang,
                        "category": category,
                        "label": label,
                        "prompt_len": len(src),
                        "comet": None
                    })
                    continue

                batch_data.append({"src": src, "mt": mt})
                meta_info.append({
                    "root_id": root_id,
                    "language": lang,
                    "category": category,
                    "label": label,
                    "prompt_len": len(src)
                })

        # ===============================
        # RUN COMET
        # ===============================
        if len(batch_data) > 0:
            scores = model.predict(
                batch_data,
                batch_size=32,
                gpus=1
            )["scores"]

            for i, score in enumerate(scores):
                meta = meta_info[i]
                rows.append({
                    "root_id": meta["root_id"],
                    "language": meta["language"],
                    "category": meta["category"],
                    "label": meta["label"],
                    "prompt_len": meta["prompt_len"],
                    "comet": score
                })

        # ===============================
        # SAVE BATCH
        # ===============================
        batch_file = os.path.join(
            OUTPUT_DIR,
            f"{file_name}_batch_{batch_idx//BATCH_SIZE}.csv"
        )

        df_batch = pd.DataFrame(rows)
        df_batch.to_csv(batch_file, index=False)

        all_rows.extend(rows)

        # ===============================
        # LOG TIME
        # ===============================
        batch_time = time.time() - batch_start
        total_processed += len(batch)

        log(f"[Batch {batch_idx//BATCH_SIZE}] "
            f"Processed: {len(batch)} items | "
            f"Time: {batch_time:.2f}s | "
            f"Total processed: {total_processed}")

# ===============================
# GLOBAL TIME
# ===============================
total_time = time.time() - start_global
log(f"\nTOTAL TIME: {total_time/60:.2f} minutes")

# ===============================
# FINAL MERGED FILE
# ===============================
df_all = pd.DataFrame(all_rows)
df_all.to_csv(EVAL_PATH, index=False)

# ===============================
# LANGUAGE SUMMARY
# ===============================
summary = {}

for lang, group in df_all.groupby("language"):
    scores = group["comet"].dropna()

    if len(scores) == 0:
        summary[lang] = {
            "mean_comet": None,
            "std_comet": None,
            "count": 0
        }
    else:
        summary[lang] = {
            "mean_comet": float(np.mean(scores)),
            "std_comet": float(np.std(scores)),
            "count": int(len(scores))
        }

with open(SUMMARY_FILE, "w") as f:
    json.dump(summary, f, indent=2)

log("Saved final summary")