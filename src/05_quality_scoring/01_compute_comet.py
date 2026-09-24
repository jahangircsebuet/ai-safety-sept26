#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute COMET-Kiwi scores for flattened multilingual AEGIS translations
(new_data pipeline).

Metric
------
COMET-Kiwi (reference-free):

    src = original English prompt
    mt  = translated target-language prompt (prompt_translated_lang)

Input
-----
Intended to run AFTER 05_1_compute_bertscore.py, on that script's output
(so rows already carry bertscore_f1). Any other fields already present on
a row (chrf_score, etc.) are passed through untouched.

Resume support
---------------
Two layers:

1. If a row's comet_score is already present in --input (e.g. inherited
   from a prior run), it is copied through without recomputation.

2. If --output already exists (this script itself was interrupted),
   rows already written there are skipped and only the remainder is
   appended.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Set, Tuple

import torch

from comet import download_model, load_from_checkpoint


# =============================================================================
# Configuration
# =============================================================================

COMET_MODEL = "Unbabel/wmt22-cometkiwi-da"
COMET_BATCH_SIZE = 32

PROCESSING_CHUNK_SIZE = 256

ROOT_FILE = Path(
    "/home/malam/projects/benchmarks/ai-safety-sept26/data/new_data/aegis_filtered/stage05_unique_roots.jsonl"
)

COMET_SUPPORTED_LANGUAGES = {
    "Afrikaans", "Albanian", "Amharic", "Arabic", "Armenian", "Assamese",
    "Azerbaijani", "Basque", "Belarusian", "Bengali", "Bosnian", "Breton",
    "Bulgarian", "Burmese", "Catalan", "Chinese (Simplified)",
    "Chinese (Traditional)", "Croatian", "Czech", "Danish", "Dutch",
    "English", "Esperanto", "Estonian", "Filipino (Tagalog)", "Finnish",
    "French", "Galician", "Georgian", "German", "Greek", "Gujarati",
    "Hausa", "Hebrew", "Hindi", "Hungarian", "Icelandic", "Indonesian",
    "Irish", "Italian", "Japanese", "Javanese", "Kannada", "Kazakh",
    "Khmer", "Korean", "Kyrgyz", "Lao", "Latvian", "Lithuanian",
    "Macedonian", "Malay", "Malayalam", "Marathi", "Mongolian", "Nepali",
    "Norwegian", "Oriya (Odia)", "Oromo", "Pashto", "Persian", "Polish",
    "Portuguese", "Punjabi", "Romanian", "Russian", "Serbian", "Sindhi",
    "Sinhala", "Slovak", "Slovenian", "Somali", "Spanish", "Swahili",
    "Swedish", "Tamil", "Telugu", "Thai", "Tosk Albanian", "Turkish",
    "Ukrainian", "Urdu", "Uzbek", "Vietnamese", "Welsh", "Xhosa",
}


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
    Rows already present in output_file with comet_score/comet_supported
    already decided (i.e. this script already finished them).
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

            if "comet_supported" in row:
                done.add(row_key(row))

    return done


# =============================================================================
# COMET with OOM-safe retry
# =============================================================================

def score_with_retry(
    comet_model,
    comet_data: List[Dict[str, str]],
    batch_size: int,
    comet_gpus: int,
) -> List[float]:

    sizes_to_try = []

    for candidate in (batch_size, max(1, batch_size // 4), max(1, batch_size // 16), 1):

        if candidate not in sizes_to_try:
            sizes_to_try.append(candidate)

    last_exc = None

    for bs in sizes_to_try:

        try:

            model_output = comet_model.predict(
                comet_data,
                batch_size=bs,
                gpus=comet_gpus,
                progress_bar=False,
            )

            return list(model_output.scores)

        except torch.cuda.OutOfMemoryError as exc:

            last_exc = exc

            torch.cuda.empty_cache()

            print(
                f"  [warning] CUDA OOM at batch_size={bs}, "
                "retrying smaller..."
            )

            continue

    raise last_exc


# =============================================================================
# Main
# =============================================================================

def compute_comet(
    input_file: Path,
    output_file: Path,
) -> None:

    print("=" * 80)
    print("COMET-KIWI (new_data pipeline)")
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

    use_cuda = torch.cuda.is_available()
    comet_gpus = 1 if use_cuda else 0

    print()
    print(f"Device: {'cuda' if use_cuda else 'cpu'}")

    print()
    print("Loading COMET model:")
    print(f"  {COMET_MODEL}")

    comet_model_path = download_model(COMET_MODEL)
    comet_model = load_from_checkpoint(comet_model_path)

    # Newer transformers releases unified XLMRobertaTokenizerFast into
    # XLMRobertaTokenizer and dropped the legacy
    # build_inputs_with_special_tokens method that comet's encoder calls
    # directly (comet/encoders/base.py: concat_sequences). Restore the
    # standard XLM-R pair-building behaviour: <s> A </s></s> B </s>.
    encoder_tokenizer = comet_model.encoder.tokenizer

    if not hasattr(encoder_tokenizer, "build_inputs_with_special_tokens"):

        def _build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):

            if token_ids_1 is None:
                return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

            cls = [self.cls_token_id]
            sep = [self.sep_token_id]

            return cls + token_ids_0 + sep + sep + token_ids_1 + sep

        import types

        encoder_tokenizer.build_inputs_with_special_tokens = types.MethodType(
            _build_inputs_with_special_tokens, encoder_tokenizer
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    file_mode = "a" if output_file.exists() else "w"

    newly_scored = 0
    inherited = 0
    skipped_existing = 0
    unsupported = 0
    total_seen = 0

    print()
    print("=" * 80)
    print("COMPUTING COMET")
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

            comet_data = []
            comet_indices = []

            for index, row in enumerate(pending_rows):

                language_name = clean_text(row.get("language_name"))

                row["comet_supported"] = (
                    language_name in COMET_SUPPORTED_LANGUAGES
                )

                # Already has a real score inherited from a previous run
                # (e.g. rows carried over from an earlier crashed run) --
                # trust it, do not recompute.
                if row.get("comet_score") is not None:
                    inherited += 1
                    continue

                if not row["comet_supported"]:
                    row["comet_score"] = None
                    unsupported += 1
                    continue

                translated_prompt = clean_text(
                    row.get("prompt_translated_lang")
                )

                if not translated_prompt:
                    row["comet_score"] = None
                    continue

                root_id = clean_text(row.get("root_id"))

                if root_id not in root_prompt_map:
                    raise KeyError(
                        "\nTranslation row references a root "
                        "that is missing from the root file:\n"
                        f"  root_id = {root_id}"
                    )

                source_prompt = root_prompt_map[root_id]

                row["comet_score"] = None

                comet_data.append({
                    "src": source_prompt,
                    "mt": translated_prompt,
                })

                comet_indices.append(index)

            if comet_data:

                scores = score_with_retry(
                    comet_model,
                    comet_data,
                    COMET_BATCH_SIZE,
                    comet_gpus,
                )

                for row_index, score in zip(comet_indices, scores):

                    pending_rows[row_index]["comet_score"] = float(score)

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
                f"unsupported={unsupported:,} | "
                f"skipped_existing={skipped_existing:,}"
            )

    print()
    print("=" * 80)
    print("COMET COMPUTATION COMPLETE")
    print("=" * 80)

    print(f"Total rows seen:              {total_seen:,}")
    print(f"Newly scored this run:        {newly_scored:,}")
    print(f"Inherited (already scored):   {inherited:,}")
    print(f"Unsupported language:         {unsupported:,}")
    print(f"Skipped (already complete):   {skipped_existing:,}")
    print()
    print(f"Saved: {output_file}")


# =============================================================================
# CLI
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compute COMET-Kiwi for flattened multilingual "
            "AEGIS translations (new_data pipeline). Resume-safe."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input JSONL, normally the output of "
            "05_1_compute_bertscore.py."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output JSONL (comet_score + comet_supported added). "
            "If it already exists, already-scored rows are skipped "
            "and new rows are appended."
        ),
    )

    args = parser.parse_args()

    compute_comet(
        input_file=Path(args.input),
        output_file=Path(args.output),
    )


if __name__ == "__main__":
    main()
