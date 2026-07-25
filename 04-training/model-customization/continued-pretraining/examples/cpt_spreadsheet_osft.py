#!/usr/bin/env python3
"""Continued pretraining on Excel spreadsheets using OSFT.

Downloads the SpreadsheetBench dataset, converts ``.xlsx`` files to markdown
text via ``markitdown``, then runs OSFT with ``is_pretraining=True`` to inject
spreadsheet understanding into an instruct model without catastrophic
forgetting.

Adapted from the Training Hub ``osft_cpt_spreadsheet_example``.

Prerequisites:
    pip install 'markitdown[xlsx]' huggingface_hub

Hardware requirements:
    2+ A100 40 GB GPUs (auto-detected by default).

Usage:
    # End-to-end: download, convert, train
    python cpt_spreadsheet_osft.py \\
        --ckpt-output-dir /path/to/checkpoints

    # Use pre-prepared data
    python cpt_spreadsheet_osft.py \\
        --data-path /path/to/spreadsheets.jsonl \\
        --ckpt-output-dir /path/to/checkpoints

    # Data preparation only
    python cpt_spreadsheet_osft.py --prepare-only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

import torch

from training_hub import osft

DEFAULT_MODEL_PATH = "ibm-granite/granite-3.3-8b-instruct"
DEFAULT_NPROC = max(torch.cuda.device_count(), 1) if torch.cuda.is_available() else 1

DATASET_REPO = "KAKA22/SpreadsheetBench"
DATASET_FILE = "spreadsheetbench_verified_400.tar.gz"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def download_dataset(cache_dir: str) -> str:
    """Download SpreadsheetBench from Hugging Face and return the archive path."""
    from huggingface_hub import hf_hub_download

    print(f"Downloading {DATASET_FILE} from {DATASET_REPO}...")
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=DATASET_FILE,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    print(f"  Downloaded to: {path}")
    return path


def extract_dataset(tar_path: str, extract_dir: str) -> str:
    """Safely extract the archive, rejecting path-traversal entries."""
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        base = Path(extract_dir).resolve()
        safe: list[tarfile.TarInfo] = []
        for member in tar.getmembers():
            target = (base / member.name).resolve()
            if os.path.commonpath([str(base), str(target)]) != str(base):
                raise ValueError(f"Unsafe tar member path: {member.name}")
            safe.append(member)
        tar.extractall(path=extract_dir, members=safe)
    print(f"  Extracted to: {extract_dir}")
    return extract_dir


def find_xlsx_files(extract_dir: str) -> list[str]:
    """Return ``_init.xlsx`` paths (input spreadsheets, not answer sheets)."""
    all_xlsx = sorted(
        glob.glob(os.path.join(extract_dir, "**", "*.xlsx"), recursive=True)
    )
    init = [f for f in all_xlsx if "_init.xlsx" in f]
    result = init or all_xlsx
    print(f"  Found {len(result)} spreadsheet files")
    return result


def convert_xlsx_to_text(xlsx_path: str) -> str | None:
    """Convert a spreadsheet to markdown text using ``markitdown``."""
    from markitdown import MarkItDown

    md = MarkItDown(enable_plugins=False)
    try:
        result = md.convert(xlsx_path)
        text = result.text_content.strip()
        return text or None
    except Exception as exc:
        print(f"  Warning: skipping {os.path.basename(xlsx_path)}: {exc}")
        return None


def prepare_data(output_jsonl: str, cache_dir: str = "./spreadsheet_cache") -> str:
    """Download, extract, convert, and write the training JSONL."""
    os.makedirs(cache_dir, exist_ok=True)
    extract_dir = os.path.join(cache_dir, "extracted")

    tar_path = download_dataset(cache_dir)
    extract_dataset(tar_path, extract_dir)

    xlsx_files = find_xlsx_files(extract_dir)
    if not xlsx_files:
        print("Error: no .xlsx files found")
        sys.exit(1)

    print(f"Converting {len(xlsx_files)} spreadsheets to text...")
    doc_count = skip_count = 0

    with open(output_jsonl, "w") as out:
        for i, path in enumerate(xlsx_files, 1):
            text = convert_xlsx_to_text(path)
            if text:
                out.write(json.dumps({"document": text}) + "\n")
                doc_count += 1
            else:
                skip_count += 1
            if i % 50 == 0:
                print(f"  Processed {i}/{len(xlsx_files)} files...")

    print(f"Data preparation complete: {doc_count} documents, {skip_count} skipped")
    if doc_count == 0:
        print("Error: no documents were successfully converted")
        sys.exit(1)
    return output_jsonl


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continued pretraining on Excel spreadsheets with OSFT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--data-path", help="Pre-prepared JSONL file (auto-prepared if omitted)")
    parser.add_argument("--ckpt-output-dir", help="Checkpoint directory (required unless --prepare-only)")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare data, skip training")
    parser.add_argument("--output-jsonl", default="./spreadsheet_pretraining_data.jsonl", help="Output JSONL path")
    parser.add_argument("--cache-dir", default="./spreadsheet_cache", help="Cache for downloaded files")

    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help=f"Model ID (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--unfreeze-rank-ratio", type=float, default=0.3, help="OSFT unfreeze ratio (default: 0.3)")
    parser.add_argument("--num-epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--block-size", type=int, default=2048, help="Document packing block size (default: 2048)")
    parser.add_argument("--nproc-per-node", type=int, default=DEFAULT_NPROC, help=f"Number of GPUs (default: {DEFAULT_NPROC})")
    parser.add_argument("--max-tokens-per-gpu", type=int, default=10000, help="Tokens per GPU per step (default: 10000)")
    parser.add_argument("--batch-size", type=int, default=64, help="Effective batch size (default: 64)")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate (default: 5e-6)")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Max sequence length (default: 4096)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.unfreeze_rank_ratio <= 1.0:
        print("Error: --unfreeze-rank-ratio must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)

    # Step 1: Prepare data
    if args.data_path and os.path.exists(args.data_path):
        data_path = args.data_path
        print(f"Using existing data: {data_path}")
    else:
        if args.data_path:
            print(f"Data not found at {args.data_path}, preparing from scratch...")
        data_path = prepare_data(output_jsonl=args.output_jsonl, cache_dir=args.cache_dir)

    if args.prepare_only:
        print("--prepare-only set, skipping training.")
        return

    # Step 2: Train
    if not args.ckpt_output_dir:
        print("Error: --ckpt-output-dir is required for training", file=sys.stderr)
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_output_dir = f"data/osft_cpt_spreadsheet_{timestamp}"

    print()
    print("=" * 60)
    print("OSFT Continued Pretraining: Excel Spreadsheets")
    print("=" * 60)
    print(f"  Model:              {args.model_path}")
    print(f"  Data:               {data_path}")
    print(f"  Output:             {args.ckpt_output_dir}")
    print(f"  GPUs:               {args.nproc_per_node}")
    print(f"  Unfreeze ratio:     {args.unfreeze_rank_ratio}")
    print(f"  Epochs:             {args.num_epochs}")
    print(f"  Block size:         {args.block_size}")
    print(f"  Batch size:         {args.batch_size}")
    print(f"  Learning rate:      {args.learning_rate}")
    print(f"  Max seq len:        {args.max_seq_len:,}")
    print(f"  Tokens / GPU:       {args.max_tokens_per_gpu:,}")
    print("=" * 60)

    start = time.time()

    try:
        osft(
            model_path=args.model_path,
            data_path=data_path,
            ckpt_output_dir=args.ckpt_output_dir,
            unfreeze_rank_ratio=args.unfreeze_rank_ratio,
            is_pretraining=True,
            block_size=args.block_size,
            document_column_name="document",
            num_epochs=args.num_epochs,
            effective_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_seq_len=args.max_seq_len,
            max_tokens_per_gpu=args.max_tokens_per_gpu,
            data_output_dir=data_output_dir,
            warmup_steps=50,
            use_liger=True,
            seed=42,
            lr_scheduler="cosine",
            checkpoint_at_epoch=True,
            save_final_checkpoint=True,
            nproc_per_node=args.nproc_per_node,
            nnodes=1,
        )

        elapsed = time.time() - start
        print("=" * 60)
        print(f"Training completed in {elapsed / 3600:.2f} hours")
        print(f"Checkpoints: {args.ckpt_output_dir}/hf_format/")
        print("=" * 60)

    except Exception as exc:
        elapsed = time.time() - start
        print(f"\nTraining failed after {elapsed / 60:.1f} minutes: {exc}", file=sys.stderr)
        print("Troubleshooting:", file=sys.stderr)
        print("  - OOM? Reduce --max-tokens-per-gpu or --block-size", file=sys.stderr)
        print("  - Run --prepare-only first to inspect the JSONL", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
