# ============================================================
# FULL FILTERING PIPELINE (BERTScore + COMET)
# Updated:
#   - Gold set
#   - High-Quality set
#   - Standard-Quality set
#   - Best per (root_id, language)
# ============================================================

import json
import numpy as np
import pandas as pd

# =========================
# FILE PATHS
# =========================
BERT_FILE = "/home/malam/projects/benchmarks/ai-safety-sept26/v1.0/data/02_quality_scores/bert_scores.csv"

COMET_FILE = "/home/malam/projects/benchmarks/ai-safety-sept26/v1.0/data/02_quality_scores/comet_scores.csv"

MERGED_OUT = "/home/malam/projects/benchmarks/ai-safety-sept26/v1.0/data/03_quality_filtered/merged_scores.csv"

GOLD_OUT = "/home/malam/projects/benchmarks/ai-safety-sept26/v1.0/data/03_quality_filtered/filtered_gold.csv"

HIGH_QUALITY_OUT = "/home/malam/projects/benchmarks/ai-safety-sept26/v1.0/data/03_quality_filtered/filtered_high_quality.csv"

STANDARD_QUALITY_OUT = "/home/malam/projects/benchmarks/ai-safety-sept26/v1.0/data/03_quality_filtered/filtered_standard_quality.csv"

BEST_OUT = "/home/malam/projects/benchmarks/ai-safety-sept26/v1.0/data/03_quality_filtered/best_per_root_language.csv"

SUMMARY_OUT = "/home/malam/projects/benchmarks/ai-safety-sept26/v1.0/data/03_quality_filtered/filtering_summary.json"

# =========================
# THRESHOLDS
# =========================

# Gold (highest quality)
GOLD_F1 = 0.95
GOLD_COMET = 0.92
GOLD_COMBINED = 0.94

# High-Quality
HIGH_QUALITY_F1 = 0.90
HIGH_QUALITY_COMET = 0.85
HIGH_QUALITY_COMBINED = 0.88

# Standard-Quality
STANDARD_QUALITY_F1 = 0.80
STANDARD_QUALITY_COMET = 0.70
STANDARD_QUALITY_COMBINED = 0.75

# =========================
# LOAD DATA
# =========================
print("Loading data...")

bert_df = pd.read_csv(BERT_FILE)
comet_df = pd.read_csv(COMET_FILE)

comet_df["comet"] = pd.to_numeric(
    comet_df["comet"],
    errors="coerce"
)

# =========================
# MERGE
# =========================
merge_keys = ["root_id", "language"]

merged = bert_df.merge(
    comet_df[merge_keys + ["comet"]],
    on=merge_keys,
    how="left"
)

print(f"Merged rows: {len(merged)}")

# =========================
# COMBINED SCORE
# =========================
merged["combined_score"] = np.where(
    merged["comet"].notna(),
    (merged["f1"] + merged["comet"]) / 2.0,
    merged["f1"]
)

merged["comet_available"] = (
    merged["comet"].notna().astype(int)
)

# =========================
# QUALITY BUCKET
# =========================
def quality_bucket(row):

    if pd.notna(row["comet"]):

        # GOLD
        if (
            row["f1"] >= GOLD_F1 and
            row["comet"] >= GOLD_COMET and
            row["combined_score"] >= GOLD_COMBINED
        ):
            return "gold"

        # HIGH QUALITY
        elif (
            row["f1"] >= HIGH_QUALITY_F1 and
            row["comet"] >= HIGH_QUALITY_COMET and
            row["combined_score"] >= HIGH_QUALITY_COMBINED
        ):
            return "high_quality"

        # STANDARD QUALITY
        elif (
            row["f1"] >= STANDARD_QUALITY_F1 and
            row["comet"] >= STANDARD_QUALITY_COMET and
            row["combined_score"] >= STANDARD_QUALITY_COMBINED
        ):
            return "standard_quality"

        else:
            return "low"

    else:

        # fallback: only F1 available

        if row["f1"] >= GOLD_F1:
            return "gold"

        elif row["f1"] >= HIGH_QUALITY_F1:
            return "high_quality"

        elif row["f1"] >= STANDARD_QUALITY_F1:
            return "standard_quality"

        else:
            return "low"


merged["quality_bucket"] = merged.apply(
    quality_bucket,
    axis=1
)

# =========================
# FILTERING
# =========================
print("Applying filters...")

# ------------------------------------------------
# GOLD
# ------------------------------------------------
gold_mask = np.where(
    merged["comet"].notna(),

    (
        (merged["f1"] >= GOLD_F1) &
        (merged["comet"] >= GOLD_COMET) &
        (merged["combined_score"] >= GOLD_COMBINED)
    ),

    (
        merged["f1"] >= GOLD_F1
    )
)

filtered_gold = merged.loc[gold_mask].copy()

# ------------------------------------------------
# HIGH QUALITY
# ------------------------------------------------
high_quality_mask = np.where(
    merged["comet"].notna(),

    (
        (merged["f1"] >= HIGH_QUALITY_F1) &
        (merged["comet"] >= HIGH_QUALITY_COMET) &
        (merged["combined_score"] >= HIGH_QUALITY_COMBINED)
    ),

    (
        merged["f1"] >= HIGH_QUALITY_F1
    )
)

filtered_high_quality = merged.loc[
    high_quality_mask
].copy()

# ------------------------------------------------
# STANDARD QUALITY
# ------------------------------------------------
standard_quality_mask = np.where(
    merged["comet"].notna(),

    (
        (merged["f1"] >= STANDARD_QUALITY_F1) &
        (merged["comet"] >= STANDARD_QUALITY_COMET) &
        (merged["combined_score"] >= STANDARD_QUALITY_COMBINED)
    ),

    (
        merged["f1"] >= STANDARD_QUALITY_F1
    )
)

filtered_standard_quality = merged.loc[
    standard_quality_mask
].copy()

# =========================
# BEST PER ROOT + LANGUAGE
# =========================
best_per_root_language = (
    merged.sort_values(
        ["root_id", "language", "combined_score"],
        ascending=[True, True, False]
    )
    .drop_duplicates(
        subset=["root_id", "language"],
        keep="first"
    )
    .copy()
)

# =========================
# SAVE FILES
# =========================
print("Saving outputs...")

merged.to_csv(MERGED_OUT, index=False)

filtered_gold.to_csv(
    GOLD_OUT,
    index=False
)

filtered_high_quality.to_csv(
    HIGH_QUALITY_OUT,
    index=False
)

filtered_standard_quality.to_csv(
    STANDARD_QUALITY_OUT,
    index=False
)

best_per_root_language.to_csv(
    BEST_OUT,
    index=False
)

# =========================
# SUMMARY
# =========================
summary = {

    "total_rows": int(len(merged)),

    "gold_kept": int(len(filtered_gold)),

    "high_quality_kept": int(
        len(filtered_high_quality)
    ),

    "standard_quality_kept": int(
        len(filtered_standard_quality)
    ),

    "gold_ratio": float(
        len(filtered_gold) / len(merged)
    ),

    "high_quality_ratio": float(
        len(filtered_high_quality) / len(merged)
    ),

    "standard_quality_ratio": float(
        len(filtered_standard_quality) / len(merged)
    ),

    "quality_bucket_counts": (
        merged["quality_bucket"]
        .value_counts()
        .to_dict()
    )
}

with open(SUMMARY_OUT, "w") as f:
    json.dump(summary, f, indent=2)

# =========================
# LOG
# =========================
print("\n===== FINAL SUMMARY =====")

print(
    f"Gold: {len(filtered_gold)} "
    f"({len(filtered_gold)/len(merged):.2%})"
)

print(
    f"High-Quality: {len(filtered_high_quality)} "
    f"({len(filtered_high_quality)/len(merged):.2%})"
)

print(
    f"Standard-Quality: {len(filtered_standard_quality)} "
    f"({len(filtered_standard_quality)/len(merged):.2%})"
)

print("\nDONE")