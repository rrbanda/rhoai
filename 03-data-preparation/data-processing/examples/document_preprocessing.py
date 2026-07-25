"""Preprocess documents into SDG-ready seed data using Docling.

Converts a document (PDF, DOCX, URL, etc.) into Markdown, chunks the text,
and writes a JSONL file with 'document' and 'domain' columns ready for SDG Hub.

Usage:
    python document_preprocessing.py --input paper.pdf --output seed_data.jsonl
"""

import argparse
import json
from pathlib import Path

from docling.document_converter import DocumentConverter


def chunk_text(text: str, chunk_size: int) -> list[str]:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert documents to SDG-ready seed JSONL.")
    p.add_argument("--input", required=True, help="URL or local file path to convert.")
    p.add_argument("--output", default="seed_data.jsonl", help="Output JSONL path.")
    p.add_argument("--chunk-size", type=int, default=512, help="Target chunk size (chars).")
    p.add_argument("--domain", default="general", help="Domain label for generated rows.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Converting: {args.input}")
    converter = DocumentConverter()
    result = converter.convert(args.input)
    markdown = result.document.export_to_markdown()
    print(f"Extracted {len(markdown)} characters of Markdown")

    chunks = chunk_text(markdown, args.chunk_size)
    print(f"Split into {len(chunks)} chunks (target size: {args.chunk_size})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        for chunk in chunks:
            record = {"document": chunk, "domain": args.domain}
            fh.write(json.dumps(record) + "\n")

    print(f"Seed dataset written to {output_path} ({len(chunks)} rows)")


if __name__ == "__main__":
    main()
