#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


# =========================================================
# 🔷 IO HELPERS
# =========================================================
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =========================================================
# 🔷 OPTIONAL HARMONIZATION
# Keep SAME object structure, only ensure missing keys exist
# =========================================================
EXPECTED_KEYS = [
    "custom_id",
    "provider",
    "model",
    "root_id",
    "language",
    "prompt",
    "system_instruction",
    "generation_config",
    "eval",
    "response",
    "response_raw",
    "provider_response_id",
    "prompt_length_tokens",
    "response_length_tokens",
    "token_count_method",
    "time_per_sample_sec",
    "finish_reason",
    "error",
    "source_request_file",
]


def harmonize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preserve the same row structure as the original batch outputs.
    Only adds missing keys as None / {} for consistency across providers.
    """
    out = dict(row)

    for key in EXPECTED_KEYS:
        if key not in out:
            if key in {"generation_config", "eval"}:
                out[key] = {}
            else:
                out[key] = None

    return out


# =========================================================
# 🔷 MAIN
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing batch response JSONL files",
    )
    parser.add_argument(
        "--out-file",
        required=True,
        help="Merged output JSONL file with SAME row structure as input rows",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip rows where error is not null",
    )
    parser.add_argument(
        "--deduplicate-by-custom-id",
        action="store_true",
        help="Drop duplicate rows based on custom_id while merging",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No provider output JSONL files found in {input_dir}")

    # If rerunning merger, remove existing output to avoid accidental double append
    if out_file.exists():
        out_file.unlink()

    seen_custom_ids = set()
    total = 0
    kept = 0
    skipped_errors = 0
    skipped_duplicates = 0

    for f in files:
        print(f"Loading provider output: {f}")
        rows = load_jsonl(f)
        out_records: List[Dict[str, Any]] = []

        for row in rows:
            total += 1

            if args.skip_errors and row.get("error") not in (None, "", {}):
                skipped_errors += 1
                continue

            cid = row.get("custom_id")
            if args.deduplicate_by_custom_id and cid is not None:
                if cid in seen_custom_ids:
                    skipped_duplicates += 1
                    continue
                seen_custom_ids.add(cid)

            out_records.append(harmonize_row(row))
            kept += 1

        if out_records:
            append_jsonl(out_file, out_records)

    print("\n✅ Merge completed")
    print(f"Total intermediate rows seen: {total}")
    print(f"Rows written: {kept}")
    print(f"Skipped errors: {skipped_errors}")
    print(f"Skipped duplicates: {skipped_duplicates}")
    print(f"Merged output saved to: {out_file}")


if __name__ == "__main__":
    main()