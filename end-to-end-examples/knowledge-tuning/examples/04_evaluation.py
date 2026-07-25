"""Step 4: Evaluate the fine-tuned model against the base model.

Loads both the base and fine-tuned models, runs inference on a set of
test questions, and prints the responses side by side for comparison.

Usage:
    python 04_evaluation.py --tuned-model ./checkpoints/final
    python 04_evaluation.py --base-model meta-llama/Llama-3.1-8B-Instruct \
                            --tuned-model ./checkpoints/final \
                            --test-questions test_questions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

FALLBACK_QUESTIONS = [
    "What are the key principles of the subject covered in the training documents?",
    "Summarize the main topics discussed in the knowledge base.",
    "What are common misconceptions about this domain?",
    "Explain the relationship between the core concepts in this area.",
    "What practical applications arise from the knowledge in the training data?",
]


def load_test_questions(path: str | None) -> list[str]:
    """Load test questions from a JSONL file, or fall back to defaults."""
    if path and Path(path).is_file():
        questions = []
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                questions.append(record["question"])
        print(f"Loaded {len(questions)} test questions from {path}")
        return questions

    if path:
        print(f"WARNING: {path} not found, using built-in fallback questions.")
    else:
        print("No test questions file specified, using built-in fallback questions.")
    return FALLBACK_QUESTIONS


def load_model_and_tokenizer(
    model_path: str, device: str
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal LM and its tokenizer."""
    print(f"Loading model: {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    max_new_tokens: int = 512,
) -> str:
    """Generate a single response for the given question."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a knowledgeable assistant. Answer the user's question "
                "accurately and thoroughly based on your training data."
            ),
        },
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def wrap_text(text: str, width: int = 76) -> str:
    """Wrap text for readable terminal output."""
    return "\n".join(textwrap.fill(line, width=width) for line in text.splitlines())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base vs fine-tuned model on test questions."
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model name or path (default: $STUDENT_MODEL)",
    )
    parser.add_argument(
        "--tuned-model",
        required=True,
        help="Path to fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--test-questions",
        default=None,
        help="JSONL file with test questions (one {'question': '...'} per line)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens to generate per response (default: 512)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for model loading (default: auto)",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    base_model_name = args.base_model or os.getenv(
        "STUDENT_MODEL", "meta-llama/Llama-3.1-8B-Instruct"
    )
    tuned_model_path = args.tuned_model

    if not Path(tuned_model_path).exists():
        print(f"ERROR: Tuned model path does not exist: {tuned_model_path}")
        sys.exit(1)

    questions = load_test_questions(args.test_questions)

    base_model, base_tokenizer = load_model_and_tokenizer(
        base_model_name, args.device
    )
    tuned_model, tuned_tokenizer = load_model_and_tokenizer(
        tuned_model_path, args.device
    )

    separator = "=" * 80
    for i, question in enumerate(questions, 1):
        print(f"\n{separator}")
        print(f"Question {i}/{len(questions)}:")
        print(f"  {question}")
        print(separator)

        base_response = generate_response(
            base_model, base_tokenizer, question, args.max_new_tokens
        )
        tuned_response = generate_response(
            tuned_model, tuned_tokenizer, question, args.max_new_tokens
        )

        print(f"\n--- Base Model ({base_model_name}) ---")
        print(wrap_text(base_response))
        print(f"\n--- Fine-Tuned Model ({tuned_model_path}) ---")
        print(wrap_text(tuned_response))

    print(f"\n{separator}")
    print(f"Evaluation complete. {len(questions)} questions processed.")
    print(separator)


if __name__ == "__main__":
    main()
