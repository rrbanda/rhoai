"""Preprocess documents into SDG-ready seed data using Docling.

Converts documents (PDF, DOCX, URL, etc.) into Markdown via Docling, chunks
the text using markdown-aware splitting that preserves document structure,
and writes a JSONL file ready for SDG Hub knowledge tuning flows.

The script supports two chunking strategies:
  - paragraph: simple paragraph-boundary splitting (fast, default)
  - markdown:  structure-aware splitting that respects headings, lists,
               tables, and code blocks (requires markdown-it-py)

It can also attach in-context learning (ICL) metadata — a document outline,
example document excerpt, and example questions — to each chunk.  This is
the format expected by SDG Hub knowledge-tuning flows.

Adapted from:
  sdg_hub/examples/knowledge_tuning/enhanced_summary_knowledge_tuning/
  document_pre_processing.ipynb

Usage:
    # Basic: convert a single PDF
    python document_preprocessing.py --input paper.pdf --output seed_data.jsonl

    # Process an entire directory of documents
    python document_preprocessing.py --input docs/ --output seed_data.jsonl

    # Markdown-aware chunking with ICL metadata
    python document_preprocessing.py \\
        --input paper.pdf --output seed_data.jsonl \\
        --chunk-strategy markdown --chunk-size 5000 --overlap 1000 \\
        --document-outline "Technical reference for ACME product" \\
        --icl-document "## Overview\\nThis guide explains..." \\
        --icl-queries "What is ACME?" "How do I install it?" "What are the requirements?"

    # With custom domain label
    python document_preprocessing.py \\
        --input paper.pdf --domain finance --output seed_data.jsonl
"""

import argparse
import glob
import json
from pathlib import Path
from typing import Optional

from docling.document_converter import DocumentConverter


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

def chunk_by_paragraphs(text: str, chunk_size: int, **_kwargs: object) -> list[str]:
    """Split *text* on paragraph boundaries into ~*chunk_size*-char segments."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > chunk_size and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_by_markdown(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    """Split markdown text at block-level elements with word-count sizing.

    Respects headings, lists, tables, code blocks, and blockquotes.
    Adds *overlap* words of context between consecutive chunks.
    """
    from markdown_it import MarkdownIt

    md = MarkdownIt()
    tokens = md.parse(text)

    blocks: list[str] = []
    buf: list[str] = []
    for tok in tokens:
        if tok.block and tok.type.endswith("_open"):
            buf = []
        elif tok.block and tok.type.endswith("_close"):
            if buf:
                blocks.append("\n".join(buf).strip())
                buf = []
        elif tok.content:
            buf.append(tok.content)
    if buf:
        blocks.append("\n".join(buf).strip())

    chunks: list[str] = []
    current_words: list[str] = []
    for block in blocks:
        words = block.split()
        for w in words:
            current_words.append(w)
            if len(current_words) >= chunk_size:
                chunks.append(" ".join(current_words))
                current_words = current_words[-overlap:] if overlap > 0 else []
    if current_words:
        chunks.append(" ".join(current_words))
    return chunks


CHUNK_STRATEGIES = {
    "paragraph": chunk_by_paragraphs,
    "markdown": chunk_by_markdown,
}


# ---------------------------------------------------------------------------
# Document conversion
# ---------------------------------------------------------------------------

def convert_document(source: str) -> str:
    """Convert a single document/URL to Markdown via Docling."""
    converter = DocumentConverter()
    result = converter.convert(source)
    return result.document.export_to_markdown()


def convert_directory(directory: str) -> list[tuple[str, str]]:
    """Convert every supported file in *directory*, returning (name, markdown) pairs."""
    supported = ("*.pdf", "*.docx", "*.doc", "*.pptx", "*.html", "*.htm")
    paths: list[str] = []
    for ext in supported:
        paths.extend(glob.glob(str(Path(directory) / ext)))
    if not paths:
        raise FileNotFoundError(
            f"No supported documents found in {directory}. "
            f"Supported extensions: {', '.join(supported)}"
        )

    results: list[tuple[str, str]] = []
    converter = DocumentConverter()
    for path in sorted(paths):
        print(f"  Converting: {path}")
        result = converter.convert(path)
        md = result.document.export_to_markdown()
        results.append((Path(path).stem, md))
    return results


# ---------------------------------------------------------------------------
# Seed data assembly
# ---------------------------------------------------------------------------

def build_seed_records(
    chunks: list[str],
    domain: str,
    document_outline: Optional[str] = None,
    icl_document: Optional[str] = None,
    icl_queries: Optional[list[str]] = None,
) -> list[dict]:
    """Build JSONL-ready records from document chunks.

    When ICL metadata is provided the output matches the schema expected by
    SDG Hub knowledge-tuning flows (document, domain, document_outline,
    icl_document, icl_query_1 … icl_query_N).
    """
    records: list[dict] = []
    for chunk in chunks:
        record: dict = {"document": chunk, "domain": domain}
        if document_outline:
            record["document_outline"] = document_outline
        if icl_document:
            record["icl_document"] = icl_document
        if icl_queries:
            for i, q in enumerate(icl_queries, 1):
                record[f"icl_query_{i}"] = q
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert documents to SDG-ready seed JSONL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--input", required=True,
        help="URL, local file path, or directory of documents to convert.",
    )
    p.add_argument("--output", default="seed_data.jsonl", help="Output JSONL path.")
    p.add_argument("--domain", default="general", help="Domain label for generated rows.")

    chunking = p.add_argument_group("chunking options")
    chunking.add_argument(
        "--chunk-strategy",
        choices=list(CHUNK_STRATEGIES),
        default="paragraph",
        help="Chunking strategy (default: paragraph).",
    )
    chunking.add_argument(
        "--chunk-size", type=int, default=512,
        help="Target chunk size — characters for 'paragraph', words for 'markdown' (default: 512).",
    )
    chunking.add_argument(
        "--overlap", type=int, default=0,
        help="Word overlap between consecutive chunks (markdown strategy only).",
    )

    icl = p.add_argument_group("ICL metadata (for knowledge tuning)")
    icl.add_argument(
        "--document-outline", type=str, default=None,
        help="High-level description of the document for ICL context.",
    )
    icl.add_argument(
        "--icl-document", type=str, default=None,
        help="Representative excerpt from the document used as an ICL example.",
    )
    icl.add_argument(
        "--icl-queries", type=str, nargs="*", default=None,
        help="Example questions based on the ICL document (up to 3 recommended).",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    chunk_fn = CHUNK_STRATEGIES[args.chunk_strategy]
    chunk_kwargs = {"chunk_size": args.chunk_size, "overlap": args.overlap}

    # --- Convert documents ---------------------------------------------------
    if input_path.is_dir():
        print(f"Processing directory: {input_path}")
        doc_pairs = convert_directory(str(input_path))
    else:
        print(f"Converting: {args.input}")
        markdown = convert_document(args.input)
        doc_pairs = [(input_path.stem, markdown)]

    # --- Chunk and build records ---------------------------------------------
    all_records: list[dict] = []
    for name, markdown in doc_pairs:
        print(f"  {name}: {len(markdown)} characters of Markdown extracted")
        chunks = chunk_fn(markdown, **chunk_kwargs)
        print(f"  {name}: split into {len(chunks)} chunks")
        records = build_seed_records(
            chunks,
            domain=args.domain,
            document_outline=args.document_outline,
            icl_document=args.icl_document,
            icl_queries=args.icl_queries,
        )
        all_records.extend(records)

    # --- Write output --------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        for record in all_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nSeed dataset written to {output_path} ({len(all_records)} rows)")


if __name__ == "__main__":
    main()
