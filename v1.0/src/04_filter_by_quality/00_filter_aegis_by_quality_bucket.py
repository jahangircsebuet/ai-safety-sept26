#!/usr/bin/env python3
"""
filter_aegis_by_quality_bucket.py

Filter aegis_multilang.json using selected:

    (root_id, language, category)

triples from filtered_gold.csv.

Matching rule
-------------

CSV:
    root_id
    language
    category

AEGIS:
    id
    translation[language]
    most_severe_category

Therefore:

    CSV root_id   == AEGIS id
    CSV language  == translation key
    CSV category  == AEGIS most_severe_category

The output contains one JSONL row per unique matched:

    (root_id, language, category)

Duplicate source occurrences of the same triple are skipped.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

try:
    import ijson
except ImportError:
    ijson = None


# ================================================================
# Helpers
# ================================================================

def clean_value(value):
    """Convert empty values to None."""
    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    return value


def to_float(value):
    """Safely convert value to float."""
    value = clean_value(value)

    if value is None:
        return None

    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def to_int(value):
    """Safely convert value to int."""
    value = clean_value(value)

    if value is None:
        return None

    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def normalize_root_id(value):
    """Normalize root ID."""
    value = clean_value(value)

    if value is None:
        return None

    return value


def normalize_language(value):
    """
    Normalize language conservatively.

    Do not lowercase because translation keys are case-sensitive.
    """
    value = clean_value(value)

    if value is None:
        return None

    return value


def normalize_category(value):
    """
    Normalize category conservatively.

    Only strip surrounding whitespace.
    """
    value = clean_value(value)

    if value is None:
        return None

    return value


# ================================================================
# Read quality CSV
# ================================================================

def load_quality_selection(csv_path):
    """
    Read filtered_gold.csv.

    Structure:

        selected[root_id][category][language] = metadata

    Therefore the effective selection key is:

        (root_id, language, category)
    """

    selected = defaultdict(
        lambda: defaultdict(dict)
    )

    total_rows = 0
    skipped_missing_keys = 0

    with open(
        csv_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise RuntimeError(
                "CSV has no header."
            )

        print("\nCSV columns:")

        for col in reader.fieldnames:
            print(f"  {repr(col)}")

        required = {
            "root_id",
            "language",
            "category",
        }

        missing = required - set(reader.fieldnames)

        if missing:
            raise RuntimeError(
                f"Required CSV columns missing: "
                f"{sorted(missing)}"
            )

        for row in reader:

            total_rows += 1

            root_id = normalize_root_id(
                row.get("root_id")
            )

            language = normalize_language(
                row.get("language")
            )

            category = normalize_category(
                row.get("category")
            )

            # All three keys are now required.
            if (
                root_id is None
                or language is None
                or category is None
            ):
                skipped_missing_keys += 1
                continue

            quality_metadata = {
                "root_id": root_id,
                "language": language,
                "category": category,

                "tier":
                    clean_value(
                        row.get("tier")
                    ),

                "f1":
                    to_float(
                        row.get("f1")
                    ),

                "prompt_len":
                    to_int(
                        row.get("prompt_len")
                    ),

                "label":
                    clean_value(
                        row.get("label")
                    ),

                "failure":
                    to_int(
                        row.get("failure")
                    ),

                "severe_failure":
                    to_int(
                        row.get("severe_failure")
                    ),

                "comet":
                    to_float(
                        row.get("comet")
                    ),

                "combined_score":
                    to_float(
                        row.get("combined_score")
                    ),

                "comet_available":
                    to_int(
                        row.get("comet_available")
                    ),

                "quality_bucket":
                    clean_value(
                        row.get("quality_bucket")
                    ),
            }

            # ----------------------------------------------------
            # Keyed by:
            #
            # root_id -> category -> language
            #
            # Duplicate CSV triples collapse automatically.
            # Last metadata row wins.
            # ----------------------------------------------------

            selected[
                root_id
            ][
                category
            ][
                language
            ] = quality_metadata


    unique_triples = sum(
        len(languages)
        for categories in selected.values()
        for languages in categories.values()
    )


    print("\n========================================")
    print("QUALITY SELECTION LOADED")
    print("========================================")

    print(
        f"CSV rows                      : "
        f"{total_rows:,}"
    )

    print(
        f"Rows missing required keys    : "
        f"{skipped_missing_keys:,}"
    )

    print(
        f"Unique root IDs               : "
        f"{len(selected):,}"
    )

    print(
        f"Unique root-language-category : "
        f"{unique_triples:,}"
    )

    return selected


# ================================================================
# Stream input JSON
# ================================================================

def iter_json_records(json_path):
    """
    Stream records from the top-level JSON array.

    Uses ijson when available.
    """

    if ijson is not None:

        print(
            "\nUsing streaming JSON reader (ijson)."
        )

        with open(json_path, "rb") as f:
            yield from ijson.items(
                f,
                "item"
            )

    else:

        print(
            "\nWARNING: ijson not installed. "
            "Loading entire JSON into memory."
        )

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        if not isinstance(data, list):
            raise RuntimeError(
                "Expected a top-level JSON list."
            )

        yield from data


# ================================================================
# Filter dataset
# ================================================================

def filter_dataset(
    json_path,
    selected,
    output_path,
    missing_path,
):
    """
    Filter using:

        (root_id, language, category)

    Category source in AEGIS:

        most_severe_category
    """

    # ------------------------------------------------------------
    # Build expected triple set from CSV.
    # ------------------------------------------------------------

    expected_triples = {
        (
            root_id,
            language,
            category,
        )

        for root_id, categories
        in selected.items()

        for category, languages
        in categories.items()

        for language
        in languages
    }


    matched_triples = set()

    # Critical:
    # prevents duplicate output rows even if source JSON
    # contains duplicate root records.
    written_triples = set()


    roots_scanned = 0
    matched_root_ids = 0

    output_rows = 0

    duplicate_source_triples_skipped = 0
    category_mismatch_roots = 0
    language_missing_count = 0


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as fout:

        for root_record in iter_json_records(
            json_path
        ):

            roots_scanned += 1

            root_id = normalize_root_id(
                root_record.get("id")
            )


            # ====================================================
            # FILTER 1 — root_id
            # ====================================================

            if root_id not in selected:
                continue

            matched_root_ids += 1


            # ====================================================
            # FILTER 2 — category
            #
            # CSV category is matched against the source field:
            #
            #     most_severe_category
            # ====================================================

            source_category = normalize_category(
                root_record.get(
                    "most_severe_category"
                )
            )


            if source_category is None:
                category_mismatch_roots += 1
                continue


            available_categories = selected[
                root_id
            ]


            if source_category not in available_categories:

                category_mismatch_roots += 1

                continue


            # ====================================================
            # Obtain source translations
            # ====================================================

            translations = root_record.get(
                "translation",
                {}
            )

            if not isinstance(
                translations,
                dict
            ):
                continue


            wanted_languages = available_categories[
                source_category
            ]


            # ----------------------------------------------------
            # Normalized lookup for source language keys.
            # ----------------------------------------------------

            normalized_language_lookup = {
                normalize_language(k): k
                for k in translations.keys()
            }


            # ====================================================
            # FILTER 3 — language
            # ====================================================

            for (
                language,
                quality_meta,
            ) in wanted_languages.items():

                translation = translations.get(
                    language
                )

                matched_language_key = language


                # ------------------------------------------------
                # Fallback:
                # whitespace-normalized language lookup.
                # ------------------------------------------------

                if translation is None:

                    actual_key = (
                        normalized_language_lookup.get(
                            language
                        )
                    )

                    if actual_key is not None:

                        translation = translations[
                            actual_key
                        ]

                        matched_language_key = (
                            actual_key
                        )


                if translation is None:

                    language_missing_count += 1
                    continue


                # =================================================
                # Triple identity
                # =================================================

                triple = (
                    root_id,
                    language,
                    source_category,
                )


                # =================================================
                # Prevent duplicate output
                # =================================================

                if triple in written_triples:

                    duplicate_source_triples_skipped += 1

                    continue


                written_triples.add(
                    triple
                )

                matched_triples.add(
                    triple
                )


                # =================================================
                # Construct output record
                # =================================================

                output_record = {

                    # ---------------------------------------------
                    # Identity
                    # ---------------------------------------------

                    "root_id":
                        root_id,

                    "language":
                        language,

                    "category":
                        source_category,


                    # ---------------------------------------------
                    # Quality metadata from filtered_gold.csv
                    # ---------------------------------------------

                    "tier":
                        quality_meta.get(
                            "tier"
                        ),

                    "f1":
                        quality_meta.get(
                            "f1"
                        ),

                    "prompt_len":
                        quality_meta.get(
                            "prompt_len"
                        ),

                    "label":
                        quality_meta.get(
                            "label"
                        ),

                    "failure":
                        quality_meta.get(
                            "failure"
                        ),

                    "severe_failure":
                        quality_meta.get(
                            "severe_failure"
                        ),

                    "comet":
                        quality_meta.get(
                            "comet"
                        ),

                    "combined_score":
                        quality_meta.get(
                            "combined_score"
                        ),

                    "comet_available":
                        quality_meta.get(
                            "comet_available"
                        ),

                    "quality_bucket":
                        quality_meta.get(
                            "quality_bucket"
                        ),


                    # ---------------------------------------------
                    # Original AEGIS data
                    # ---------------------------------------------

                    "id":
                        root_record.get(
                            "id"
                        ),

                    "reconstruction_id_if_redacted":
                        root_record.get(
                            "reconstruction_id_if_redacted"
                        ),

                    "prompt_original":
                        root_record.get(
                            "prompt"
                        ),

                    "response_original":
                        root_record.get(
                            "response"
                        ),

                    "prompt_label":
                        root_record.get(
                            "prompt_label"
                        ),

                    "response_label":
                        root_record.get(
                            "response_label"
                        ),

                    "violated_categories":
                        root_record.get(
                            "violated_categories"
                        ),

                    "most_severe_category":
                        root_record.get(
                            "most_severe_category"
                        ),

                    "prompt_label_source":
                        root_record.get(
                            "prompt_label_source"
                        ),

                    "response_label_source":
                        root_record.get(
                            "response_label_source"
                        ),


                    # ---------------------------------------------
                    # Selected multilingual translation
                    # ---------------------------------------------

                    "prompt_translated":
                        translation.get(
                            "prompt_translated_lang"
                        ),

                    "prompt_backtranslated":
                        translation.get(
                            "prompt_back_to_original_lang"
                        ),

                    "response_translated":
                        translation.get(
                            "response_translated_lang"
                        ),

                    "response_backtranslated":
                        translation.get(
                            "response_back_to_original_lang"
                        ),

                    "translation_model":
                        translation.get(
                            "model_name"
                        ),

                    "translation_language_key":
                        matched_language_key,
                }


                fout.write(
                    json.dumps(
                        output_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                output_rows += 1


    # ============================================================
    # Missing triples
    # ============================================================

    missing_triples = sorted(
        expected_triples
        - matched_triples
    )


    with open(
        missing_path,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "root_id",
            "language",
            "category",
        ])

        writer.writerows(
            missing_triples
        )


    # ============================================================
    # Summary
    # ============================================================

    print("\n========================================")
    print("FILTERING SUMMARY")
    print("========================================")

    print(
        f"AEGIS records scanned                 : "
        f"{roots_scanned:,}"
    )

    print(
        f"Source records with selected root_id  : "
        f"{matched_root_ids:,}"
    )

    print(
        f"Expected unique triples               : "
        f"{len(expected_triples):,}"
    )

    print(
        f"Matched unique triples                : "
        f"{len(matched_triples):,}"
    )

    print(
        f"Missing triples                       : "
        f"{len(missing_triples):,}"
    )

    print(
        f"Output rows written                   : "
        f"{output_rows:,}"
    )

    print(
        f"Duplicate source triples skipped      : "
        f"{duplicate_source_triples_skipped:,}"
    )

    print(
        f"Source category mismatches            : "
        f"{category_mismatch_roots:,}"
    )

    print(
        f"Selected languages missing in source  : "
        f"{language_missing_count:,}"
    )


    if expected_triples:

        coverage = (
            len(matched_triples)
            / len(expected_triples)
            * 100.0
        )

        print(
            f"Triple coverage                       : "
            f"{coverage:.2f}%"
        )


    print(
        f"\nFiltered output:\n"
        f"  {output_path}"
    )

    print(
        f"\nMissing triples report:\n"
        f"  {missing_path}"
    )


# ================================================================
# CLI
# ================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Filter aegis_multilang.json using "
            "(root_id, language, category) triples "
            "from filtered_gold.csv."
        )
    )


    parser.add_argument(
        "--multilang-json",
        type=Path,
        required=True,
        help="Path to aegis_multilang.json",
    )


    parser.add_argument(
        "--quality-csv",
        type=Path,
        required=True,
        help="Path to filtered_gold.csv",
    )


    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file",
    )


    parser.add_argument(
        "--missing-output",
        type=Path,
        default=None,
        help=(
            "Optional CSV containing selected "
            "(root_id, language, category) triples "
            "that could not be found."
        ),
    )


    args = parser.parse_args()


    if args.missing_output is None:

        missing_output = (
            args.output.parent
            / (
                f"{args.output.stem}"
                "_missing_triples.csv"
            )
        )

    else:

        missing_output = (
            args.missing_output
        )


    print(
        "\nLoading quality selection..."
    )

    selected = load_quality_selection(
        args.quality_csv
    )


    print(
        "\nFiltering multilingual dataset..."
    )

    filter_dataset(
        json_path=
            args.multilang_json,

        selected=
            selected,

        output_path=
            args.output,

        missing_path=
            missing_output,
    )


if __name__ == "__main__":
    main()