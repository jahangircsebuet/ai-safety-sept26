#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute BERTScore F1 for flattened multilingual AEGIS translations
(new_data pipeline).

Metric
------
BERTScore F1 between:

    original English prompt  <->  English backtranslation
    (prompt_back_to_original_lang)

Input
-----
Flattened translation JSONL, one row per root_id x language. Rows may
already carry other metric fields (comet_score, chrf_score, ...) from a
prior stage or a previous partial run -- those are passed through
untouched.

Resume support
---------------
If --output already exists, rows already scored (bertscore_f1 not null,
matched by (root_id, language_name)) are skipped. Only missing rows are
computed and appended to the same file.

Numerical stability
--------------------
Runs in plain fp32. Mixed precision (both fp16 and bf16 autocast) was
tried and abandoned: DeBERTa's attention-mask fill computes
torch.finfo(query_layer.dtype).min, but query_layer stays fp32 for this
op under autocast while attention_scores is cast down, so the fp32
sentinel value overflows the narrower dtype on write (this happens for
fp16 AND bf16, since bf16's max magnitude is very slightly smaller than
fp32's despite the same exponent range).

If a batch still hits CUDA OOM in fp32, it is retried at progressively
smaller sub-batch sizes (down to 1) before giving up on that row.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Set, Tuple

import torch

from bert_score import BERTScorer


# =============================================================================
# Configuration
# =============================================================================

BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"
BERTSCORE_BATCH_SIZE = 32

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
    Rows already present in output_file with bertscore_f1 computed.
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

            if row.get("bertscore_f1") is not None:
                done.add(row_key(row))

    return done


# =============================================================================
# BERTScore with OOM-safe retry
# =============================================================================

def score_with_retry(
    bert_scorer: BERTScorer,
    cands: List[str],
    refs: List[str],
    batch_size: int,
) -> List[float]:
    """
    Score in plain fp32 (mixed precision abandoned -- both fp16 and bf16
    autocast hit a DeBERTa attention-mask-fill overflow bug for this
    model). On CUDA OOM, clear the cache and retry at progressively
    smaller sub-batch sizes (helps when a single unusually long text
    blows up memory).
    """

    sizes_to_try = []

    for candidate in (batch_size, max(1, batch_size // 4), max(1, batch_size // 16), 1):

        if candidate not in sizes_to_try:
            sizes_to_try.append(candidate)

    last_exc_message = None

    for bs in sizes_to_try:

        try:

            _, _, f1_scores = bert_scorer.score(
                cands=cands,
                refs=refs,
                batch_size=bs,
            )

            return (
                f1_scores
                .detach()
                .float()
                .cpu()
                .tolist()
            )

        except torch.cuda.OutOfMemoryError as exc:

            # Keep only the message, not the exception object itself --
            # the exception's traceback holds references to every local
            # variable (including large GPU tensors) live in every frame
            # at the moment of the crash. Holding the exception across
            # retry iterations pins that memory and empty_cache() cannot
            # reclaim it, since it is still referenced from Python. This
            # caused a real memory leak across repeated retries.
            last_exc_message = str(exc)

            torch.cuda.empty_cache()

            print(
                f"  [warning] CUDA OOM at batch_size={bs}, "
                "retrying smaller..."
            )

            continue

    raise torch.cuda.OutOfMemoryError(last_exc_message)


# =============================================================================
# Main
# =============================================================================

def compute_bertscore(
    input_file: Path,
    output_file: Path,
) -> None:

    print("=" * 80)
    print("BERTSCORE F1 (new_data pipeline)")
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
    device = "cuda" if use_cuda else "cpu"

    print()
    print(f"Device: {device}")

    if use_cuda:
        print("GPU: " + torch.cuda.get_device_name(0))

    print()
    print("Loading BERTScore model:")
    print(f"  {BERTSCORE_MODEL}  (fp32)")

    bert_scorer = BERTScorer(
        model_type=BERTSCORE_MODEL,
        lang="en",
        rescale_with_baseline=True,
        device=device,
    )

    # microsoft/deberta-xlarge-mnli ships with no configured tokenizer
    # max length, so transformers falls back to its "unset" sentinel
    # (~1e30). Newer tokenizers' Rust truncation setter can't hold that
    # in a machine word and raises OverflowError. Pin it to the model's
    # real max_position_embeddings (512) instead.
    bert_scorer._tokenizer.model_max_length = 512

    output_file.parent.mkdir(parents=True, exist_ok=True)

    file_mode = "a" if output_file.exists() else "w"

    newly_scored = 0
    skipped_existing = 0
    missing_backtranslation = 0
    total_seen = 0

    print()
    print("=" * 80)
    print("COMPUTING BERTSCORE")
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

            cand_texts = []
            ref_texts = []
            score_indices = []

            for index, row in enumerate(pending_rows):

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

                row["bertscore_f1"] = None

                if backtranslation:

                    cand_texts.append(backtranslation)
                    ref_texts.append(source_prompt)
                    score_indices.append(index)

                else:
                    missing_backtranslation += 1

            if cand_texts:

                scores = score_with_retry(
                    bert_scorer,
                    cand_texts,
                    ref_texts,
                    BERTSCORE_BATCH_SIZE,
                )

                for row_index, score in zip(score_indices, scores):

                    pending_rows[row_index]["bertscore_f1"] = float(score)

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
                f"skipped_existing={skipped_existing:,}"
            )

    print()
    print("=" * 80)
    print("BERTSCORE COMPUTATION COMPLETE")
    print("=" * 80)

    print(f"Total rows seen:              {total_seen:,}")
    print(f"Newly scored this run:        {newly_scored:,}")
    print(f"Skipped (already complete):   {skipped_existing:,}")
    print(f"Missing backtranslations:     {missing_backtranslation:,}")
    print()
    print(f"Saved: {output_file}")


# =============================================================================
# CLI
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compute BERTScore F1 for flattened multilingual "
            "AEGIS translations (new_data pipeline). Resume-safe."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input flattened translation JSONL.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output JSONL (bertscore_f1 added). "
            "If it already exists, already-scored rows are skipped "
            "and new rows are appended."
        ),
    )

    args = parser.parse_args()

    compute_bertscore(
        input_file=Path(args.input),
        output_file=Path(args.output),
    )


if __name__ == "__main__":
    main()
