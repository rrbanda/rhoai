"""Mix and sample generated knowledge data for training.

This script takes the output of SDG Hub knowledge generation flows
(extractive_summary, detailed_summary, key_facts, document_based QA) and
produces balanced training mixes at configurable "cut sizes" — the number
of summary variants sampled per source document.

Key operations:
  1. Load multiple summary-type datasets from a generation output folder
  2. Validate which cut sizes are feasible given the data
  3. Sample per-document QA pairs with configurable limits
  4. Format into chat-style messages with metadata
  5. Compute tokenization statistics using the student model's tokenizer
  6. Save combined training-ready JSONL files

Adapted from:
  sdg_hub/examples/knowledge_tuning/enhanced_summary_knowledge_tuning/
  knowledge_mixing.ipynb  &  knowledge_mixing_utils.py

Usage:
    python knowledge_mixing.py \\
        --data-dir ./generated_output_data \\
        --student-model meta-llama/Llama-3.1-8B-Instruct \\
        --cut-sizes 10 20 --qa-per-doc 3

    # With a .env file
    echo 'OUTPUT_DATA_FOLDER=./generated_output_data' > .env
    echo 'STUDENT_MODEL=meta-llama/Llama-3.1-8B-Instruct' >> .env
    python knowledge_mixing.py
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import polars as pl
from datasets import Dataset, concatenate_datasets
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

SUMMARY_TYPES_SFT = [
    "extractive_summary",
    "detailed_summary",
    "key_facts_to_qa",
    "document_based_qa",
]

SUMMARY_TYPES_CPT = [
    "extractive_summary_cpt",
    "detailed_summary_cpt",
    "key_facts_cpt",
    "document_based_cpt",
]


# ---------------------------------------------------------------------------
# Sampling helpers (ported from knowledge_mixing_utils.py)
# ---------------------------------------------------------------------------

def get_avg_summaries_per_raw_doc(df: pl.DataFrame) -> float:
    """Average number of unique summaries per raw document."""
    counts = df.group_by("raw_document").agg(
        pl.col("document").n_unique().alias("n")
    )
    return counts["n"].mean()


def sample_doc_qa(
    df: pl.DataFrame, n_docs_per_raw: int = 50, qa_per_doc: int = 3
) -> pl.DataFrame:
    """Sample summary variants per raw doc, then limit QA pairs per summary."""
    required = ["question", "response", "document", "raw_document", "document_outline"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    avg = get_avg_summaries_per_raw_doc(df)
    if avg < n_docs_per_raw:
        print(f"  Warning: cut {n_docs_per_raw} > avg summaries ({avg:.1f})")

    df = df.with_columns([pl.struct(["question", "response"]).alias("qa_pair")])

    agg_cols = [
        pl.col("qa_pair"),
        pl.col("raw_document").first(),
        pl.col("document_outline").first(),
    ]
    if "parse_response_dict_reasoning_content" in df.columns:
        df = df.with_columns(
            pl.col("parse_response_dict_reasoning_content").alias("reasoning")
        )
        agg_cols.append(pl.col("reasoning").first())

    df = df.group_by("document").agg(agg_cols)
    sampled = df.group_by("raw_document").map_groups(
        lambda g: g.sample(n=min(n_docs_per_raw, g.height))
    )
    sampled = sampled.with_columns(
        pl.col("qa_pair").list.slice(0, qa_per_doc)
    ).explode("qa_pair")
    sampled = sampled.with_columns([
        pl.col("qa_pair").struct.field("question").alias("question"),
        pl.col("qa_pair").struct.field("response").alias("response"),
    ]).drop("qa_pair")
    return sampled


def sample_docs(df: pl.DataFrame, n_docs_per_raw: int = 50) -> pl.DataFrame:
    """Sample unique summaries per raw document (CPT path)."""
    df_unique = df.group_by("document").agg([
        pl.col("raw_document").first().alias("raw_document"),
        pl.col("document_outline").first().alias("document_outline"),
    ])
    return df_unique.group_by("raw_document").map_groups(
        lambda g: g.sample(n=min(n_docs_per_raw, g.height))
    )


# ---------------------------------------------------------------------------
# Message formatting
# ---------------------------------------------------------------------------

def _make_messages(record: dict, *, with_doc: bool = True) -> list[dict]:
    """Build a user/assistant message pair from a QA record."""
    if with_doc:
        user_content = (
            f"{record['document_outline']}\n{record['document']}\n\n"
            f"{record['question']}"
        )
    else:
        user_content = f"In {record['document_outline']}, {record['question']}"

    resp = (
        record["response"]
        .replace("[ANSWER]", "")
        .replace("[END]", "")
        .strip()
    )
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": resp},
    ]


def generate_knowledge_qa_dataset(
    df: pl.DataFrame,
    keep_columns: list[str] | None = None,
    keep_document_in_context: bool = True,
) -> pl.DataFrame:
    """Convert sampled QA data into chat-message format with metadata."""
    keep_columns = keep_columns or []
    msg_cols = ["question", "response", "document", "document_outline"]

    messages_expr = (
        pl.struct(msg_cols)
        .map_elements(
            lambda r: _make_messages(r, with_doc=keep_document_in_context),
        )
        .alias("messages")
    )
    metadata_expr = (
        pl.struct([
            pl.col("document").alias("sdg_document"),
            pl.lit("document_knowledge_qa").alias("dataset"),
            pl.col("raw_document"),
        ])
        .map_elements(json.dumps)
        .alias("metadata")
    )

    result = df.with_columns([messages_expr, metadata_expr])
    result = result.select(keep_columns + ["messages", "metadata"])
    result = result.with_columns(pl.lit(True).alias("unmask"))
    return result


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_tokens(df: pl.DataFrame, tokenizer: Any, column: str = "messages") -> pl.DataFrame:
    """Add a ``token_length`` column counting tokens via *tokenizer*."""
    def _apply_template(msgs: list[dict]) -> str:
        return tokenizer.apply_chat_template(msgs, tokenize=False)

    def _count(text: str) -> int:
        return len(tokenizer.encode(text))

    if column == "messages":
        expr = (
            pl.col(column)
            .map_elements(_apply_template, return_dtype=pl.String)
            .map_elements(_count, return_dtype=pl.Int32)
            .alias("token_length")
        )
    else:
        expr = (
            pl.col(column)
            .map_elements(_count, return_dtype=pl.Int32)
            .alias("token_length")
        )
    return df.with_columns(expr)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_summary_dataset(
    data_dir: str, summary_type: str, filter_gpt_oss: bool = False
) -> pl.DataFrame | None:
    """Load a single summary-type dataset from *data_dir*."""
    path = Path(data_dir) / summary_type
    if not path.exists():
        print(f"  Skipping {summary_type} (not found at {path})")
        return None

    from datasets import load_dataset
    ds = load_dataset("json", data_dir=str(path), split="train")
    if summary_type == "document_based_qa":
        ds = ds.rename_column("base_document", "raw_document")

    if filter_gpt_oss:
        orig = len(ds)
        ds = ds.filter(
            lambda x: "..." not in x["question"]
            and "<question>" not in x["question"]
            and "<Insert question here>" not in x["question"]
        )
        ds = ds.map(lambda x: {
            "response": x["response"].replace("[ANSWER]", "").replace("[END]", "").strip()
        })
        print(f"  {summary_type}: filtered {orig - len(ds)} rows")

    print(f"  Loaded {summary_type}: {len(ds)} samples")
    return ds.to_polars()


# ---------------------------------------------------------------------------
# Core mixing logic
# ---------------------------------------------------------------------------

def process_cut(
    cut: int,
    datasets: dict[str, pl.DataFrame],
    tokenizer: Any,
    qa_per_doc: int,
    is_cpt: bool,
) -> list[Dataset]:
    """Process all summary types for a single cut size, returning HF datasets."""
    hf_datasets: list[Dataset] = []

    for stype, df in datasets.items():
        print(f"  {stype} ...")
        if stype in ("key_facts_to_qa", "key_facts_cpt"):
            result = generate_knowledge_qa_dataset(
                df,
                keep_columns=["question", "document_outline", "raw_document", "document"],
                keep_document_in_context=False,
            )
            col = "messages"
        elif is_cpt and stype in ("extractive_summary_cpt", "detailed_summary_cpt"):
            result = sample_docs(df, n_docs_per_raw=cut)
            col = "document"
        elif is_cpt:
            cols = ["document", "raw_document", "document_outline"]
            cols = [c for c in cols if c in df.columns]
            result = df.select(cols)
            col = "document"
        else:
            if stype != "document_based_qa":
                df_cut = sample_doc_qa(df, n_docs_per_raw=cut, qa_per_doc=qa_per_doc)
            else:
                df_cut = df
            result = generate_knowledge_qa_dataset(
                df_cut,
                keep_columns=["question", "document_outline", "raw_document", "document"],
                keep_document_in_context=True,
            )
            col = "messages"

        result = count_tokens(result, tokenizer, column=col)
        hf_ds = Dataset.from_polars(result)
        n_unique = len(set(hf_ds["document"])) if "document" in hf_ds.column_names else "?"
        total_tok = sum(hf_ds["token_length"])
        print(f"    {len(hf_ds)} samples, {n_unique} unique docs, {total_tok:,} tokens")
        hf_datasets.append(hf_ds)

    return hf_datasets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mix and sample generated knowledge data for training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data-dir", type=str,
        default=os.getenv("OUTPUT_DATA_FOLDER", "generated_output_data"),
        help="Folder containing per-summary-type subdirectories.",
    )
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: <data-dir>/training_mix).",
    )
    p.add_argument(
        "--student-model", type=str,
        default=os.getenv("STUDENT_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        help="HuggingFace model used for tokenization statistics.",
    )
    p.add_argument(
        "--cut-sizes", type=int, nargs="+",
        default=[int(x) for x in os.getenv("CUT_SIZES", "10,20").split(",")],
        help="Number of summary variants to sample per raw document.",
    )
    p.add_argument(
        "--qa-per-doc", type=int,
        default=int(os.getenv("QA_PER_DOC", "3")),
        help="Max QA pairs per summary (default: 3).",
    )
    p.add_argument(
        "--cpt", action="store_true",
        default=os.getenv("GENERATE_CPT", "false").lower() in ("1", "true", "yes"),
        help="Use CPT summary types instead of SFT types.",
    )
    p.add_argument(
        "--filter-gpt-oss", action="store_true",
        default=os.getenv("SAVE_GPT_OSS_FORMAT", "false").lower() == "true",
        help="Apply GPT-OSS quality filtering to loaded datasets.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir or os.path.join(args.data_dir, "training_mix"))
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_types = SUMMARY_TYPES_CPT if args.cpt else SUMMARY_TYPES_SFT

    print(f"Data directory : {args.data_dir}")
    print(f"Student model  : {args.student_model}")
    print(f"Cut sizes      : {args.cut_sizes}")
    print(f"QA per doc     : {args.qa_per_doc}")
    print(f"Summary types  : {summary_types}")
    print()

    # --- Load tokenizer and datasets ----------------------------------------
    print(f"Loading tokenizer: {args.student_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)

    datasets: dict[str, pl.DataFrame] = {}
    for stype in summary_types:
        df = load_summary_dataset(args.data_dir, stype, args.filter_gpt_oss)
        if df is not None:
            datasets[stype] = df

    if not datasets:
        raise SystemExit("No datasets loaded — check --data-dir path and contents.")

    print(f"\nLoaded {len(datasets)} summary datasets\n")

    # --- Validate feasible cut sizes ----------------------------------------
    feasible_cuts = list(args.cut_sizes)
    for stype, df in datasets.items():
        if stype in ("key_facts_to_qa", "document_based_qa", "key_facts_cpt", "document_based_cpt"):
            continue
        avg = get_avg_summaries_per_raw_doc(df)
        for cut in list(feasible_cuts):
            if avg < cut:
                print(f"Cut {cut} infeasible for {stype} (avg={avg:.1f}), removing")
                feasible_cuts.remove(cut)
    feasible_cuts = sorted(set(feasible_cuts))

    if not feasible_cuts:
        raise SystemExit("No feasible cut sizes. Reduce --cut-sizes or generate more data.")

    print(f"Feasible cuts: {feasible_cuts}\n")

    # --- Process each cut size ----------------------------------------------
    summary_rows: list[tuple[int, int, int]] = []
    for cut in feasible_cuts:
        print(f"--- Cut size {cut} ---")
        hf_datasets = process_cut(cut, datasets, tokenizer, args.qa_per_doc, args.cpt)
        if not hf_datasets:
            print(f"  No data produced for cut {cut}")
            continue

        combined = concatenate_datasets(hf_datasets)
        total_tokens = sum(combined["token_length"])
        out_path = output_dir / f"combined_cut_{cut}x.jsonl"
        combined.to_json(str(out_path), orient="records", lines=True)
        print(f"  Saved {len(combined)} samples ({total_tokens:,} tokens) -> {out_path}\n")
        summary_rows.append((cut, len(combined), total_tokens))

    # --- Final summary -------------------------------------------------------
    if summary_rows:
        print("=" * 55)
        print(f"{'Cut':>6}  {'Samples':>10}  {'Tokens':>14}")
        print("-" * 55)
        for cut, samples, tokens in summary_rows:
            print(f"{cut:>6}  {samples:>10,}  {tokens:>14,}")
        print("=" * 55)
    else:
        print("No training mixes were produced.")


if __name__ == "__main__":
    main()
