#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute chrF++ for flattened multilingual AEGIS translations
(new_data pipeline).

Metric
------
chrF++ (character n-gram F-score, word_order=2) between:

    original English prompt  <->  English backtranslation
    (prompt_back_to_original_lang)

Native scale: 0-100.

Input
-----
Intended to run AFTER 05_2_compute_comet.py, on that script's output
(so rows already carry bertscore_f1, comet_score, comet_supported). Any
other fields already present on a row are passed through untouched.

CPU-only metric -- no GPU / OOM concerns, but still resume-safe for
consistency with the other two stages and for very large inputs.

Resume support
---------------
Two layers:

1. If a row's chrf_score is already present in --input (e.g. inherited
   from a prior run), it is copied through without recomputation.

2. If --output already exists (this script itself was interrupted),
   rows already written there are skipped and only the remainder is
   appended.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Set, Tuple

from sacrebleu.metrics import CHRF


# =============================================================================
# Configuration
# =============================================================================

PROCESSING_CHUNK_SIZE = 256

ROOT_FILE = Path(
    "/home/malam/projects/benchmarks/ai-safety-sept26/data/new_data/aegis_filtered/stage05_unique_roots.jsonl"
)


# =============================================================================
# Generic helpers
# =============================================================================

def clean_text(value: Any) -> str:

    if value is None:
        return ""

    return str(value).strip()


def load_json_or_jsonl(
    path: Path,
) -> List[Dict[str, Any]]:

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            return [data]

    except json.JSONDecodeError:
        pass

    rows = []

    with path.open("r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                )

            if not isinstance(row, dict):
                raise ValueError(
                    f"Line {line_number} of {path} is not a JSON object."
                )

            rows.append(row)

    return rows


def iter_translation_rows(
    path: Path,
) -> Iterator[Dict[str, Any]]:

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as f:

        first_char = ""

        while True:

            char = f.read(1)

            if not char:
                break

            if not char.isspace():
                first_char = char
                break

    if first_char == "[":

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for row in data:

            if not isinstance(row, dict):
                raise ValueError(
                    "Input JSON array contains a non-object item."
                )

            yield row

        return

    with path.open("r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on input line {line_number}: {exc}"
                )

            if not isinstance(row, dict):
                raise ValueError(
                    f"Input line {line_number} is not a JSON object."
                )

            yield row


def load_root_prompt_map(
    root_file: Path,
) -> Dict[str, str]:

    rows = load_json_or_jsonl(root_file)

    mapping = {}

    for row in rows:

        root_id = clean_text(row.get("root_id"))
        prompt = clean_text(row.get("prompt"))

        if not root_id:
            continue

        if not prompt:
            raise ValueError(f"Missing English prompt for root_id={root_id}")

        if root_id in mapping:

            if mapping[root_id] != prompt:
                raise ValueError(
                    f"Root table contains conflicting prompts for {root_id}"
                )

            continue

        mapping[root_id] = prompt

    return mapping


def chunks(
    iterator: Iterable[Dict[str, Any]],
    chunk_size: int,
) -> Iterator[List[Dict[str, Any]]]:

    chunk = []

    for row in iterator:

        chunk.append(row)

        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []

    if chunk:
        yield chunk


def row_key(row: Dict[str, Any]) -> Tuple[str, str]:

    return (
        clean_text(row.get("root_id")),
        clean_text(row.get("language_name")),
    )


# =============================================================================
# Resume support
# =============================================================================

def load_done_keys(
    output_file: Path,
) -> Set[Tuple[str, str]]:
    """
    Rows already present in output_file with chrf_score computed.
    """

    done: Set[Tuple[str, str]] = set()

    if not output_file.exists():
        return done

    with output_file.open("r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            if row.get("chrf_score") is not None:
                done.add(row_key(row))

    return done


# =============================================================================
# Main
# =============================================================================

def compute_chrf(
    input_file: Path,
    output_file: Path,
) -> None:

    print("=" * 80)
    print("chrF++ (new_data pipeline)")
    print("=" * 80)

    print(f"Input : {input_file}")
    print(f"Output: {output_file}")
    print(f"Roots : {ROOT_FILE}")

    if not ROOT_FILE.exists():
        raise FileNotFoundError(f"Root file not found: {ROOT_FILE}")

    print()
    print("Loading canonical English root prompts...")

    root_prompt_map = load_root_prompt_map(ROOT_FILE)

    print(f"Loaded roots: {len(root_prompt_map):,}")

    print()
    print("Checking output for already-completed rows (resume)...")

    done_keys = load_done_keys(output_file)

    print(f"Already complete: {len(done_keys):,}")

    chrf_metric = CHRF(
        char_order=6,
        word_order=2,   # chrF++
        beta=2,
        lowercase=False,
        whitespace=False,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    file_mode = "a" if output_file.exists() else "w"

    newly_scored = 0
    inherited = 0
    skipped_existing = 0
    missing_backtranslation = 0
    total_seen = 0

    print()
    print("=" * 80)
    print("COMPUTING chrF++")
    print("=" * 80)

    with output_file.open(file_mode, encoding="utf-8") as output_handle:

        for chunk_number, chunk in enumerate(
            chunks(iter_translation_rows(input_file), PROCESSING_CHUNK_SIZE),
            start=1,
        ):

            pending_rows = []

            for row in chunk:

                total_seen += 1

                if row_key(row) in done_keys:
                    skipped_existing += 1
                    continue

                pending_rows.append(row)

            if not pending_rows:
                continue

            for row in pending_rows:

                # Already has a real score inherited from a previous run.
                if row.get("chrf_score") is not None:
                    inherited += 1
                    continue

                root_id = clean_text(row.get("root_id"))

                if root_id not in root_prompt_map:
                    raise KeyError(
                        "\nTranslation row references a root "
                        "that is missing from the root file:\n"
                        f"  root_id = {root_id}"
                    )

                source_prompt = root_prompt_map[root_id]

                backtranslation = clean_text(
                    row.get("prompt_back_to_original_lang")
                )

                if not backtranslation:
                    row["chrf_score"] = None
                    missing_backtranslation += 1
                    continue

                result = chrf_metric.sentence_score(
                    hypothesis=backtranslation,
                    references=[source_prompt],
                )

                row["chrf_score"] = float(result.score)

                newly_scored += 1

            for row in pending_rows:

                output_handle.write(
                    json.dumps(row, ensure_ascii=False)
                )

                output_handle.write("\n")

            output_handle.flush()

            print(
                f"[Chunk {chunk_number:05d}] "
                f"seen={total_seen:,} | "
                f"newly_scored={newly_scored:,} | "
                f"inherited={inherited:,} | "
                f"skipped_existing={skipped_existing:,}"
            )

    print()
    print("=" * 80)
    print("chrF++ COMPUTATION COMPLETE")
    print("=" * 80)

    print(f"Total rows seen:              {total_seen:,}")
    print(f"Newly scored this run:        {newly_scored:,}")
    print(f"Inherited (already scored):   {inherited:,}")
    print(f"Missing backtranslations:     {missing_backtranslation:,}")
    print(f"Skipped (already complete):   {skipped_existing:,}")
    print()
    print(f"Saved: {output_file}")


# =============================================================================
# CLI
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compute chrF++ for flattened multilingual "
            "AEGIS translations (new_data pipeline). Resume-safe."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input JSONL, normally the output of "
            "05_2_compute_comet.py."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output JSONL (chrf_score added). "
            "If it already exists, already-scored rows are skipped "
            "and new rows are appended."
        ),
    )

    args = parser.parse_args()

    compute_chrf(
        input_file=Path(args.input),
        output_file=Path(args.output),
    )


if __name__ == "__main__":
    main()
