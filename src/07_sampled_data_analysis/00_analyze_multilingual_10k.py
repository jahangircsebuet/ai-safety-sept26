#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze a multilingual benchmark JSONL and generate:
- CSV tables
- PNG plots
- one HTML report
- optional metadata-enriched JSONL

The script enriches rows from language_metadata.py using nllb_code.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_no}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{line_no}"
                )
            rows.append(row)
    return rows


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def count_table(series: pd.Series, total: int, name: str) -> pd.DataFrame:
    s = (
        series.fillna("Unknown")
        .astype(str)
        .replace("", "Unknown")
    )
    counts = s.value_counts()
    out = counts.rename_axis(name).reset_index(name="count")
    out["percent"] = (
        100.0 * out["count"] / total
        if total else 0.0
    )
    return out


# ---------------------------------------------------------------------
# Language metadata
# ---------------------------------------------------------------------

def import_metadata_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        "benchmark_language_metadata",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import metadata file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_metadata_entries(module) -> Dict[str, Dict[str, Any]]:
    merged = {}

    for variable_name, value in vars(module).items():
        if variable_name.startswith("__") or not isinstance(value, dict):
            continue

        for key, record in value.items():
            if isinstance(record, dict) and record.get("nllb_code"):
                merged.setdefault(str(key), record)

    if not merged:
        raise RuntimeError(
            "Could not find a language metadata dictionary containing nllb_code."
        )

    return merged


def first_present(record: Dict[str, Any], keys: List[str]):
    for key in keys:
        if key in record:
            value = record.get(key)
            if value is not None and normalize_text(value):
                return value
    return None


JOSHI_KEYS = [
    "joshi_class",
    "joshi_tier",
    "resource_level_joshi",
    "resource_tier_joshi",
    "joshi_resource_level",
    "joshi_resource_class",
]

GENERIC_TIER_KEYS = [
    "resource_tier",
    "tier",
    "resource_level",
]

NLLB_TIER_KEYS = [
    "resource_level_nllb",
    "nllb_resource_level",
    "nllb_tier",
]

DIRECTION_KEYS = [
    "writing_direction",
    "direction",
    "text_direction",
]


def build_metadata_by_code(entries: Dict[str, Dict[str, Any]]):
    by_code = {}

    for metadata_key, record in entries.items():
        code = normalize_text(record.get("nllb_code"))
        if not code:
            continue

        joshi = first_present(record, JOSHI_KEYS)
        generic_tier = first_present(record, GENERIC_TIER_KEYS)
        nllb_tier = first_present(record, NLLB_TIER_KEYS)

        analysis_tier = (
            joshi
            if joshi is not None
            else generic_tier
            if generic_tier is not None
            else nllb_tier
        )

        by_code[code] = {
            "metadata_key": metadata_key,
            "metadata_language": first_present(
                record, ["language", "language_name", "name"]
            ),
            "iso639_3": first_present(
                record, ["iso639_3", "iso_639_3"]
            ),
            "script_iso": first_present(
                record, ["script_iso", "script_code"]
            ),
            "script": first_present(
                record, ["script", "script_name"]
            ),
            "family": first_present(
                record, ["family", "language_family"]
            ),
            "subgrouping": first_present(
                record, ["subgrouping", "subgroup", "branch"]
            ),
            "writing_direction": first_present(
                record, DIRECTION_KEYS
            ),
            "joshi_class": joshi,
            "generic_resource_tier": generic_tier,
            "nllb_resource_level": nllb_tier,
            "analysis_tier": analysis_tier,
            "web_mt_support_2022": record.get("web_mt_support_2022"),
            "flores200_new": record.get("flores200_new"),
        }

    return by_code


def enrich_rows(rows, metadata_by_code):
    enriched = []
    stats = Counter()

    metadata_fields = [
        "metadata_key",
        "metadata_language",
        "iso639_3",
        "script_iso",
        "script",
        "family",
        "subgrouping",
        "writing_direction",
        "joshi_class",
        "generic_resource_tier",
        "nllb_resource_level",
        "analysis_tier",
        "web_mt_support_2022",
        "flores200_new",
    ]

    for row in rows:
        out = dict(row)
        code = normalize_text(row.get("nllb_code"))
        meta = metadata_by_code.get(code)

        if meta is None:
            stats["rows_without_language_metadata"] += 1
            for field in metadata_fields:
                out[field] = None
        else:
            stats["rows_with_language_metadata"] += 1
            out.update(meta)

        enriched.append(out)

    return enriched, dict(stats)


# ---------------------------------------------------------------------
# DataFrame preparation
# ---------------------------------------------------------------------

def to_dataframe(rows):
    df = pd.DataFrame(rows)

    defaults = {
        "root_id": "",
        "sample_id": "",
        "category": "Unknown",
        "language_name": "Unknown",
        "nllb_code": "Unknown",
        "sample_origin": "Unknown",
        "analysis_tier": "Unknown",
        "joshi_class": "Unknown",
        "nllb_resource_level": "Unknown",
        "family": "Unknown",
        "script": "Unknown",
        "writing_direction": "Unknown",
        "bertscore_f1": np.nan,
        "comet_score": np.nan,
        "chrf_score": np.nan,
        "comet_supported": False,
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    text_cols = [
        "category",
        "language_name",
        "nllb_code",
        "sample_origin",
        "analysis_tier",
        "joshi_class",
        "nllb_resource_level",
        "family",
        "script",
        "writing_direction",
    ]

    for col in text_cols:
        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
            .replace("", "Unknown")
        )

    for col in [
        "bertscore_f1",
        "comet_score",
        "chrf_score",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    return df


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def save_barh(
    table,
    label_col,
    value_col,
    title,
    xlabel,
    output,
    figsize=(10, 7),
    max_rows=None,
):
    plot_df = table.copy()

    if max_rows is not None:
        plot_df = plot_df.head(max_rows)

    plot_df = plot_df.iloc[::-1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(
        plot_df[label_col].astype(str),
        plot_df[value_col],
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)

    for i, value in enumerate(plot_df[value_col]):
        ax.text(
            value,
            i,
            f" {int(value):,}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_histogram(values, title, xlabel, output, bins=40):
    values = values.dropna()
    if values.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Samples")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(matrix, title, output):
    if matrix.empty:
        return

    fig_width = max(10, 1.2 * len(matrix.columns) + 6)
    fig_height = max(7, 0.42 * len(matrix.index) + 3)

    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height)
    )

    image = ax.imshow(
        matrix.to_numpy(dtype=float),
        aspect="auto",
    )

    ax.set_title(title)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(
        matrix.columns,
        rotation=45,
        ha="right",
    )
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    fig.colorbar(
        image,
        ax=ax,
        label="Samples",
    )

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_category_origin_plot(category_origin, output):
    if category_origin.empty:
        return

    matrix = category_origin.set_index("category")
    matrix = matrix.loc[
        matrix.sum(axis=1).sort_values(
            ascending=False
        ).index
    ]

    fig_height = max(
        7,
        0.42 * len(matrix.index) + 3,
    )

    fig, ax = plt.subplots(
        figsize=(11, fig_height)
    )

    matrix.iloc[::-1].plot(
        kind="barh",
        stacked=True,
        ax=ax,
    )

    ax.set_title("Category Distribution by Sample Origin")
    ax.set_xlabel("Samples")
    ax.set_ylabel("")
    ax.legend(title="Sample origin")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_boxplot_by_tier(df, metric, output):
    usable = df[
        (df["analysis_tier"] != "Unknown")
        & df[metric].notna()
    ]

    if usable.empty:
        return

    order = (
        usable["analysis_tier"]
        .value_counts()
        .index
        .tolist()
    )

    data = [
        usable.loc[
            usable["analysis_tier"] == tier,
            metric,
        ].to_numpy()
        for tier in order
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        data,
        tick_labels=order,
        showfliers=False,
    )
    ax.set_title(f"{metric} by Resource Tier")
    ax.set_xlabel("Resource tier")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_metric_scatter(df, output, sample_size=5000, seed=42):
    usable = df[
        df["bertscore_f1"].notna()
        & df["chrf_score"].notna()
    ].copy()

    if usable.empty:
        return

    if len(usable) > sample_size:
        usable = usable.sample(
            n=sample_size,
            random_state=seed,
        )

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        usable["bertscore_f1"],
        usable["chrf_score"],
        s=12,
        alpha=0.45,
    )
    ax.set_title("BERTScore F1 vs. chrF")
    ax.set_xlabel("BERTScore F1")
    ax.set_ylabel("chrF")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Analysis tables
# ---------------------------------------------------------------------

def make_quality_summary(df):
    rows = []

    for metric in [
        "bertscore_f1",
        "comet_score",
        "chrf_score",
    ]:
        values = df[metric].dropna()
        if values.empty:
            continue

        rows.append({
            "metric": metric,
            "count": int(values.count()),
            "missing": int(df[metric].isna().sum()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "p05": float(values.quantile(0.05)),
            "p25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "p75": float(values.quantile(0.75)),
            "p95": float(values.quantile(0.95)),
            "max": float(values.max()),
        })

    return pd.DataFrame(rows)


def make_group_quality_table(df, group_col):
    rows = []

    for group_value, group_df in df.groupby(
        group_col,
        dropna=False,
    ):
        row = {
            group_col: (
                "Unknown"
                if pd.isna(group_value)
                else str(group_value)
            ),
            "samples": len(group_df),
        }

        for metric in [
            "bertscore_f1",
            "comet_score",
            "chrf_score",
        ]:
            values = group_df[metric].dropna()
            row[f"{metric}_n"] = int(values.count())
            row[f"{metric}_mean"] = (
                float(values.mean())
                if len(values)
                else np.nan
            )
            row[f"{metric}_median"] = (
                float(values.median())
                if len(values)
                else np.nan
            )

        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            "samples",
            ascending=False,
        )
    return out


def category_language_coverage(df):
    rows = []

    for category, group in df.groupby("category"):
        language_counts = group["nllb_code"].value_counts()

        rows.append({
            "category": category,
            "samples": len(group),
            "unique_languages": int(group["nllb_code"].nunique()),
            "unique_roots": int(group["root_id"].nunique()),
            "largest_language_count": int(
                language_counts.max()
                if len(language_counts)
                else 0
            ),
            "smallest_language_count": int(
                language_counts.min()
                if len(language_counts)
                else 0
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            "samples",
            ascending=False,
        )
    )


def root_diversity_by_category(df):
    rows = []

    for category, group in df.groupby("category"):
        root_counts = group["root_id"].value_counts()
        unique_roots = int(group["root_id"].nunique())

        rows.append({
            "category": category,
            "samples": len(group),
            "unique_roots": unique_roots,
            "samples_per_unique_root": (
                len(group) / unique_roots
                if unique_roots
                else np.nan
            ),
            "max_languages_for_single_root": int(
                root_counts.max()
                if len(root_counts)
                else 0
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            "samples",
            ascending=False,
        )
    )


# ---------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------

def dataframe_html(df, max_rows=150):
    show = df.head(max_rows).copy()

    for col in show.select_dtypes(
        include=[np.number]
    ).columns:
        if pd.api.types.is_float_dtype(
            show[col]
        ):
            show[col] = show[col].map(
                lambda x: (
                    ""
                    if pd.isna(x)
                    else f"{x:.4f}"
                )
            )

    return show.to_html(
        index=False,
        escape=True,
        border=0,
        classes="data-table",
    )


def generate_html_report(
    report_path,
    summary,
    metadata_coverage,
    tables,
    figure_paths,
):
    css = """
<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    max-width: 1500px;
    margin: 28px auto;
    padding: 0 24px 60px 24px;
    background: #fafafa;
    color: #222;
    line-height: 1.5;
}
h1, h2, h3 { margin-top: 28px; }
.card, .figure {
    background: white;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 18px 20px;
    margin: 18px 0;
}
.figure img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0 auto;
}
.data-table {
    border-collapse: collapse;
    width: 100%;
    background: white;
    font-size: 13px;
}
.data-table th, .data-table td {
    border-bottom: 1px solid #e6e6e6;
    padding: 7px 9px;
    text-align: right;
}
.data-table th:first-child,
.data-table td:first-child {
    text-align: left;
}
.data-table th { background: #eee; }
.table-wrap {
    overflow-x: auto;
    margin-bottom: 24px;
}
.note {
    color: #555;
    font-size: 14px;
}
</style>
"""

    parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='UTF-8'>",
        "<title>10K Multilingual Benchmark Analysis</title>",
        css,
        "</head><body>",
        "<h1>10K Multilingual Benchmark Analysis</h1>",
        "<div class='card'>",
        "<h2>Dataset summary</h2>",
        dataframe_html(summary),
        "</div>",
        "<div class='card'>",
        "<h2>Language metadata coverage</h2>",
        dataframe_html(metadata_coverage),
        (
            "<p class='note'>analysis_tier prefers a Joshi/resource tier "
            "when available and otherwise falls back to the NLLB resource "
            "level. Joshi classes are reported separately.</p>"
        ),
        "</div>",
    ]

    for title, image_path in figure_paths:
        if image_path.exists():
            relative = str(
                image_path.relative_to(
                    report_path.parent
                )
            ).replace("\\", "/")

            parts.extend([
                "<div class='figure'>",
                f"<h2>{html.escape(title)}</h2>",
                (
                    f"<img src='{html.escape(relative)}' "
                    f"alt='{html.escape(title)}'>"
                ),
                "</div>",
            ])

    parts.append("<h2>Tabular analyses</h2>")

    for title, table in tables.items():
        parts.extend([
            f"<h3>{html.escape(title)}</h3>",
            "<div class='table-wrap'>",
            dataframe_html(table),
            "</div>",
        ])

    parts.extend([
        "</body></html>",
    ])

    report_path.write_text(
        "\n".join(parts),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate plots, CSV tables, and an HTML report for "
            "the final multilingual sampled JSONL."
        )
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--language-metadata-file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--top-languages",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--top-families",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--scatter-sample",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--write-enriched-jsonl",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()

    if not args.input_file.exists():
        raise FileNotFoundError(
            f"Input JSONL not found: {args.input_file}"
        )

    if not args.language_metadata_file.exists():
        raise FileNotFoundError(
            f"Language metadata file not found: {args.language_metadata_file}"
        )

    figures_dir = args.output_dir / "figures"
    tables_dir = args.output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    rows = load_jsonl(args.input_file)
    print(f"Rows loaded: {len(rows):,}")

    metadata_module = import_metadata_module(
        args.language_metadata_file
    )
    metadata_entries = discover_metadata_entries(
        metadata_module
    )
    metadata_by_code = build_metadata_by_code(
        metadata_entries
    )

    print(
        f"Metadata NLLB codes discovered: "
        f"{len(metadata_by_code):,}"
    )

    enriched_rows, metadata_stats = enrich_rows(
        rows,
        metadata_by_code,
    )

    if args.write_enriched_jsonl:
        enriched_path = (
            args.output_dir
            / "enriched_10k.jsonl"
        )
        write_jsonl(
            enriched_rows,
            enriched_path,
        )
        print(f"Enriched JSONL: {enriched_path}")

    df = to_dataframe(enriched_rows)
    total = len(df)

    # Summary.
    duplicate_sample_ids = int(
        df["sample_id"].duplicated().sum()
    )
    duplicate_root_language = int(
        df.duplicated(
            subset=["root_id", "nllb_code"]
        ).sum()
    )
    comet_supported = (
        df["comet_supported"]
        .fillna(False)
        .astype(bool)
    )

    summary = pd.DataFrame([
        {"metric": "Rows", "value": total},
        {"metric": "Unique sample_id", "value": int(df["sample_id"].nunique())},
        {"metric": "Unique root_id", "value": int(df["root_id"].nunique())},
        {"metric": "Unique languages", "value": int(df["nllb_code"].nunique())},
        {"metric": "Unique categories", "value": int(df["category"].nunique())},
        {"metric": "Duplicate sample_id rows", "value": duplicate_sample_ids},
        {"metric": "Duplicate (root_id, nllb_code) rows", "value": duplicate_root_language},
        {"metric": "COMET-supported samples", "value": int(comet_supported.sum())},
        {"metric": "COMET-unsupported samples", "value": int((~comet_supported).sum())},
    ])
    save_table(summary, tables_dir / "dataset_summary.csv")

    rows_with_meta = metadata_stats.get(
        "rows_with_language_metadata",
        0,
    )
    rows_without_meta = metadata_stats.get(
        "rows_without_language_metadata",
        0,
    )

    metadata_coverage = pd.DataFrame([
        {
            "metric": "Rows with language metadata",
            "count": rows_with_meta,
            "percent": pct(rows_with_meta, total),
        },
        {
            "metric": "Rows without language metadata",
            "count": rows_without_meta,
            "percent": pct(rows_without_meta, total),
        },
        {
            "metric": "Samples with Joshi class",
            "count": int((df["joshi_class"] != "Unknown").sum()),
            "percent": pct(int((df["joshi_class"] != "Unknown").sum()), total),
        },
        {
            "metric": "Samples with NLLB resource level",
            "count": int((df["nllb_resource_level"] != "Unknown").sum()),
            "percent": pct(int((df["nllb_resource_level"] != "Unknown").sum()), total),
        },
        {
            "metric": "Samples with analysis tier",
            "count": int((df["analysis_tier"] != "Unknown").sum()),
            "percent": pct(int((df["analysis_tier"] != "Unknown").sum()), total),
        },
    ])
    save_table(
        metadata_coverage,
        tables_dir / "metadata_coverage.csv",
    )

    # Distributions.
    category_table = count_table(
        df["category"], total, "category"
    )
    language_table = count_table(
        df["nllb_code"], total, "nllb_code"
    )

    language_name_map = (
        df.groupby("nllb_code")["language_name"]
        .agg(
            lambda x: (
                x.value_counts().index[0]
                if len(x)
                else "Unknown"
            )
        )
        .to_dict()
    )

    language_table.insert(
        1,
        "language_name",
        language_table["nllb_code"].map(
            language_name_map
        ),
    )

    tier_table = count_table(
        df["analysis_tier"], total, "analysis_tier"
    )
    joshi_table = count_table(
        df["joshi_class"], total, "joshi_class"
    )
    nllb_resource_table = count_table(
        df["nllb_resource_level"],
        total,
        "nllb_resource_level",
    )
    family_table = count_table(
        df["family"], total, "family"
    )
    script_table = count_table(
        df["script"], total, "script"
    )
    direction_table = count_table(
        df["writing_direction"],
        total,
        "writing_direction",
    )
    origin_table = count_table(
        df["sample_origin"],
        total,
        "sample_origin",
    )

    distribution_tables = {
        "category_distribution.csv": category_table,
        "language_distribution.csv": language_table,
        "tier_distribution.csv": tier_table,
        "joshi_class_distribution.csv": joshi_table,
        "nllb_resource_distribution.csv": nllb_resource_table,
        "family_distribution.csv": family_table,
        "script_distribution.csv": script_table,
        "writing_direction_distribution.csv": direction_table,
        "sample_origin_distribution.csv": origin_table,
    }

    for filename, table in distribution_tables.items():
        save_table(table, tables_dir / filename)

    # Cross-tabs.
    category_by_origin = (
        pd.crosstab(
            df["category"],
            df["sample_origin"],
        )
        .reset_index()
    )

    language_by_origin = (
        pd.crosstab(
            df["nllb_code"],
            df["sample_origin"],
        )
        .reset_index()
    )

    category_by_tier_matrix = pd.crosstab(
        df["category"],
        df["analysis_tier"],
    )

    category_by_joshi_matrix = pd.crosstab(
        df["category"],
        df["joshi_class"],
    )

    save_table(
        category_by_origin,
        tables_dir / "category_by_origin.csv",
    )
    save_table(
        language_by_origin,
        tables_dir / "language_by_origin.csv",
    )
    category_by_tier_matrix.to_csv(
        tables_dir / "category_by_tier.csv"
    )
    category_by_joshi_matrix.to_csv(
        tables_dir / "category_by_joshi_class.csv"
    )

    # Quality analyses.
    quality_summary = make_quality_summary(df)
    quality_by_origin = make_group_quality_table(
        df, "sample_origin"
    )
    quality_by_category = make_group_quality_table(
        df, "category"
    )
    quality_by_tier = make_group_quality_table(
        df, "analysis_tier"
    )

    save_table(
        quality_summary,
        tables_dir / "quality_summary.csv",
    )
    save_table(
        quality_by_origin,
        tables_dir / "quality_by_origin.csv",
    )
    save_table(
        quality_by_category,
        tables_dir / "quality_by_category.csv",
    )
    save_table(
        quality_by_tier,
        tables_dir / "quality_by_tier.csv",
    )

    category_lang_cov = category_language_coverage(df)
    root_diversity = root_diversity_by_category(df)

    save_table(
        category_lang_cov,
        tables_dir / "category_language_coverage.csv",
    )
    save_table(
        root_diversity,
        tables_dir / "root_diversity_by_category.csv",
    )

    # Figures.
    figure_paths = []

    def fig(title, filename):
        path = figures_dir / filename
        figure_paths.append((title, path))
        return path

    save_barh(
        category_table,
        "category",
        "count",
        "Category Distribution",
        "Samples",
        fig(
            "Category Distribution",
            "01_category_distribution.png",
        ),
        figsize=(11, 9),
    )

    language_plot = language_table.copy()
    language_plot["label"] = (
        language_plot["language_name"]
        + " ("
        + language_plot["nllb_code"]
        + ")"
    )

    save_barh(
        language_plot,
        "label",
        "count",
        f"Language Distribution — Top {args.top_languages}",
        "Samples",
        fig(
            "Language Distribution",
            "02_language_distribution.png",
        ),
        figsize=(
            11,
            max(8, args.top_languages * 0.28),
        ),
        max_rows=args.top_languages,
    )

    save_barh(
        tier_table,
        "analysis_tier",
        "count",
        "Resource Tier Distribution",
        "Samples",
        fig(
            "Resource Tier Distribution",
            "03_tier_distribution.png",
        ),
        figsize=(9, 6),
    )

    save_barh(
        joshi_table,
        "joshi_class",
        "count",
        "Joshi Class Distribution",
        "Samples",
        fig(
            "Joshi Class Distribution",
            "04_joshi_class_distribution.png",
        ),
        figsize=(9, 6),
    )

    save_barh(
        family_table,
        "family",
        "count",
        f"Language Family Distribution — Top {args.top_families}",
        "Samples",
        fig(
            "Language Family Distribution",
            "05_family_distribution.png",
        ),
        figsize=(
            10,
            max(7, args.top_families * 0.3),
        ),
        max_rows=args.top_families,
    )

    save_barh(
        script_table,
        "script",
        "count",
        "Script Distribution",
        "Samples",
        fig(
            "Script Distribution",
            "06_script_distribution.png",
        ),
        figsize=(10, 7),
    )

    save_barh(
        direction_table,
        "writing_direction",
        "count",
        "Writing Direction Distribution",
        "Samples",
        fig(
            "Writing Direction Distribution",
            "07_writing_direction_distribution.png",
        ),
        figsize=(8, 5),
    )

    save_barh(
        origin_table,
        "sample_origin",
        "count",
        "Sample Origin Distribution",
        "Samples",
        fig(
            "Sample Origin Distribution",
            "08_sample_origin_distribution.png",
        ),
        figsize=(8, 5),
    )

    save_category_origin_plot(
        category_by_origin,
        fig(
            "Category Distribution by Sample Origin",
            "09_category_by_origin.png",
        ),
    )

    save_heatmap(
        category_by_tier_matrix,
        "Category × Resource Tier",
        fig(
            "Category × Resource Tier",
            "10_category_by_tier_heatmap.png",
        ),
    )

    save_barh(
        category_lang_cov,
        "category",
        "unique_languages",
        "Language Coverage per Category",
        "Unique languages",
        fig(
            "Language Coverage per Category",
            "11_category_language_coverage.png",
        ),
        figsize=(11, 9),
    )

    save_histogram(
        df["bertscore_f1"],
        "BERTScore F1 Distribution",
        "BERTScore F1",
        fig(
            "BERTScore F1 Distribution",
            "12_bertscore_distribution.png",
        ),
    )

    save_histogram(
        df["comet_score"],
        "COMET Score Distribution",
        "COMET score",
        fig(
            "COMET Score Distribution",
            "13_comet_distribution.png",
        ),
    )

    save_histogram(
        df["chrf_score"],
        "chrF Score Distribution",
        "chrF score",
        fig(
            "chrF Score Distribution",
            "14_chrf_distribution.png",
        ),
    )

    save_boxplot_by_tier(
        df,
        "bertscore_f1",
        fig(
            "BERTScore F1 by Resource Tier",
            "15_quality_by_tier_bertscore.png",
        ),
    )

    save_metric_scatter(
        df,
        fig(
            "BERTScore F1 vs. chrF",
            "16_quality_metric_correlation.png",
        ),
        sample_size=args.scatter_sample,
        seed=args.seed,
    )

    html_tables = {
        "Category distribution": category_table,
        "Language distribution": language_table,
        "Resource-tier distribution": tier_table,
        "Joshi-class distribution": joshi_table,
        "NLLB resource distribution": nllb_resource_table,
        "Sample-origin distribution": origin_table,
        "Quality summary": quality_summary,
        "Quality by origin": quality_by_origin,
        "Quality by category": quality_by_category,
        "Quality by tier": quality_by_tier,
        "Category-language coverage": category_lang_cov,
        "Root diversity by category": root_diversity,
    }

    report_path = (
        args.output_dir
        / "analysis_report.html"
    )

    generate_html_report(
        report_path,
        summary,
        metadata_coverage,
        html_tables,
        figure_paths,
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Rows:             {total:,}")
    print(f"Unique roots:     {df['root_id'].nunique():,}")
    print(f"Languages:        {df['nllb_code'].nunique():,}")
    print(f"Categories:       {df['category'].nunique():,}")
    print(f"HTML report:      {report_path}")
    print(f"Tables directory: {tables_dir}")
    print(f"Figures directory:{figures_dir}")


if __name__ == "__main__":
    main()