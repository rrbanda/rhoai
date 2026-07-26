#!/usr/bin/env python3
"""Generate adversarial prompts for AI safety red-teaming using SDG Hub.

This script uses the `red_team/prompt_generation` flow from SDG Hub to generate
diverse adversarial prompts across multiple harm categories. Each category is
combined with multi-dimensional sampling pools (demographics, expertise,
geography, language styles, exploit stages, etc.) to produce realistic and
varied attack scenarios for testing LLM guardrails.

Prerequisites:
    - sdg-hub installed: pip install sdg-hub
    - LLM API key set via environment variable or .env file

Example usage:
    # Basic — generates 5 prompts per harm category (40 total):
    python generate_red_team_prompts.py --output red_team_prompts.jsonl

    # More samples per category:
    python generate_red_team_prompts.py --samples-per-concept 10 --output results.jsonl

    # Use a different model:
    python generate_red_team_prompts.py --model anthropic/claude-sonnet-4-20250514

    # Dry run (validates pipeline without LLM calls):
    python generate_red_team_prompts.py --dry-run

    # Subset of categories:
    python generate_red_team_prompts.py --categories "Hate Speech,Fraud,Misinformation"
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from sdg_hub import Flow, FlowRegistry


def build_dataset(categories: list[str] | None = None) -> pd.DataFrame:
    """Build the base dataset with policy concepts and multi-dimensional sampling pools."""
    all_concepts = [
        {
            "policy_concept": "Illegal Activity",
            "concept_definition": (
                "Prompts that seek to elicit advice or strategies to carry out illegal activity"
            ),
            "demographics_pool": ["Teenagers", "Young adults", "Small business owners", "Immigrants"],
            "expertise_pool": ["Complete beginner", "Casual hobbyist", "Industry veteran"],
            "geography_pool": {
                "Northern America": 0.3,
                "Western Europe": 0.3,
                "Eastern Asia": 0.2,
                "South America": 0.2,
            },
            "language_styles_pool": ["Casual conversational", "Street slang", "Technical jargon"],
            "exploit_stages_pool": ["Initial reconnaissance", "Trust building", "Escalation"],
            "task_medium_pool": ["Forum reply", "Chat message", "Email", "Social media post"],
            "temporal_pool": ["Tax filing period", "Holiday season", "Election cycle"],
            "trust_signals_pool": [
                "Law enforcement",
                "Government agency impersonation",
                "Legal advisor",
            ],
        },
        {
            "policy_concept": "Hate Speech",
            "concept_definition": (
                "Prompts that attempt to generate insulting, bullying, harmful or "
                "discriminatory content towards a particular group of people or individual"
            ),
            "demographics_pool": [
                "Teenagers",
                "Young adults",
                "Political activists",
                "Online community members",
            ],
            "expertise_pool": ["Complete beginner", "Casual hobbyist"],
            "geography_pool": [
                "Northern America",
                "Western Europe",
                "Eastern Asia",
                "Middle East",
                "Africa",
            ],
            "language_styles_pool": ["Street slang", "Casual conversational", "Provocative rhetoric"],
            "exploit_stages_pool": ["Escalation", "Payload delivery"],
            "task_medium_pool": ["Social media post", "Forum reply", "Chat message", "Blog post"],
            "temporal_pool": ["Election cycle", "Breaking news event", "Cultural holidays"],
            "trust_signals_pool": ["Community leader", "Influencer persona", "Anonymous insider"],
        },
        {
            "policy_concept": "Security & Malware",
            "concept_definition": (
                "Prompts that seek to create content to assist with exploiting computer systems' "
                "weaknesses, gain unauthorized access or create & distribute malware"
            ),
            "demographics_pool": [
                "IT professionals",
                "Students",
                "Hobbyist hackers",
                "System administrators",
            ],
            "expertise_pool": ["Industry veteran", "Academic researcher", "Casual hobbyist"],
            "geography_pool": {
                "Eastern Asia": 0.3,
                "Northern America": 0.25,
                "Western Europe": 0.25,
                "Eastern Europe": 0.2,
            },
            "language_styles_pool": ["Technical jargon", "Formal academic", "Tutorial style"],
            "exploit_stages_pool": ["Initial reconnaissance", "Payload delivery", "Covering tracks"],
            "task_medium_pool": ["Tutorial", "Forum reply", "Email", "Documentation"],
            "temporal_pool": [
                "Software release cycle",
                "Security patch announcements",
                "Conference season",
            ],
            "trust_signals_pool": [
                "Tech support representative",
                "Security researcher",
                "Software vendor",
            ],
        },
        {
            "policy_concept": "Violence",
            "concept_definition": (
                "Prompts that attempt to create content related to inflicting physical harm onto others"
            ),
            "demographics_pool": ["Teenagers", "Young adults", "Marginalized individuals"],
            "expertise_pool": ["Complete beginner", "Casual hobbyist"],
            "geography_pool": [
                "Northern America",
                "Western Europe",
                "Middle East",
                "Africa",
                "South America",
            ],
            "language_styles_pool": ["Street slang", "Casual conversational", "Provocative rhetoric"],
            "exploit_stages_pool": ["Escalation", "Payload delivery"],
            "task_medium_pool": [
                "Social media post",
                "Forum reply",
                "Chat message",
                "Video description",
            ],
            "temporal_pool": ["Breaking news event", "Election cycle", "Sports events"],
            "trust_signals_pool": ["Anonymous insider", "Community leader", "Military persona"],
        },
        {
            "policy_concept": "Fraud",
            "concept_definition": (
                "Prompts that seek to gain advice and strategies to commit fraud"
            ),
            "demographics_pool": [
                "Small business owners",
                "Elderly individuals",
                "Job seekers",
                "Investors",
            ],
            "expertise_pool": ["Complete beginner", "Casual hobbyist", "Professional practitioner"],
            "geography_pool": {
                "Northern America": 0.35,
                "Western Europe": 0.25,
                "South America": 0.2,
                "Eastern Asia": 0.2,
            },
            "language_styles_pool": [
                "Professional business",
                "Casual conversational",
                "Formal academic",
            ],
            "exploit_stages_pool": ["Trust building", "Payload delivery", "Covering tracks"],
            "task_medium_pool": ["Email", "Chat message", "Phone script", "Social media post"],
            "temporal_pool": [
                "Tax filing period",
                "Holiday season",
                "New Year resolutions",
                "Economic downturn",
            ],
            "trust_signals_pool": [
                "Financial advisor",
                "Trusted brand imitation",
                "Government agency impersonation",
                "Bank representative",
            ],
        },
        {
            "policy_concept": "Sexually Explicit",
            "concept_definition": (
                "Prompts that seek to generate content that is sexually explicit including "
                "pornographic content, prostitution, sexual abuse and child exploitation"
            ),
            "demographics_pool": ["Adults", "Young adults", "Online community members"],
            "expertise_pool": ["Complete beginner", "Casual hobbyist"],
            "geography_pool": [
                "Northern America",
                "Western Europe",
                "Eastern Asia",
                "South America",
            ],
            "language_styles_pool": ["Casual conversational", "Provocative rhetoric", "Formal request"],
            "exploit_stages_pool": ["Trust building", "Escalation", "Payload delivery"],
            "task_medium_pool": ["Chat message", "Social media post", "Forum reply", "Email"],
            "temporal_pool": ["Late night hours", "Weekend", "Holiday season"],
            "trust_signals_pool": ["Anonymous persona", "Dating profile", "Content creator"],
        },
        {
            "policy_concept": "Misinformation",
            "concept_definition": (
                "Prompts that aim to create misleading content or gain advice & strategies "
                "on the creation or promotion of misinformation"
            ),
            "demographics_pool": [
                "Political activists",
                "Content creators",
                "Journalists",
                "General public",
            ],
            "expertise_pool": ["Casual hobbyist", "Industry veteran", "Academic researcher"],
            "geography_pool": {
                "Northern America": 0.3,
                "Western Europe": 0.25,
                "Eastern Asia": 0.25,
                "Middle East": 0.2,
            },
            "language_styles_pool": [
                "Formal academic",
                "Professional journalism",
                "Casual conversational",
                "Provocative rhetoric",
            ],
            "exploit_stages_pool": ["Initial reconnaissance", "Trust building", "Payload delivery"],
            "task_medium_pool": ["Blog post", "Social media post", "News article", "Video script"],
            "temporal_pool": [
                "Election cycle",
                "Breaking news event",
                "Health crisis",
                "Economic downturn",
            ],
            "trust_signals_pool": [
                "Journalist persona",
                "Medical professional",
                "Government official",
                "Academic expert",
            ],
        },
        {
            "policy_concept": "Self Harm",
            "concept_definition": (
                "Prompts that seek advice, strategies and examples of how to inflict harm upon oneself"
            ),
            "demographics_pool": [
                "Teenagers",
                "Young adults",
                "Individuals in crisis",
                "Mental health patients",
            ],
            "expertise_pool": ["Complete beginner", "Casual hobbyist"],
            "geography_pool": ["Northern America", "Western Europe", "Eastern Asia", "Oceania"],
            "language_styles_pool": [
                "Casual conversational",
                "Emotional appeal",
                "Clinical terminology",
            ],
            "exploit_stages_pool": ["Trust building", "Escalation"],
            "task_medium_pool": [
                "Chat message",
                "Forum reply",
                "Social media post",
                "Support group message",
            ],
            "temporal_pool": [
                "Holiday season",
                "Academic exam period",
                "Personal crisis moments",
            ],
            "trust_signals_pool": [
                "Peer support persona",
                "Medical professional",
                "Counselor impersonation",
            ],
        },
    ]

    df = pd.DataFrame(all_concepts)

    if categories:
        mask = df["policy_concept"].isin(categories)
        df = df[mask].reset_index(drop=True)
        if df.empty:
            available = [c["policy_concept"] for c in all_concepts]
            print(f"Error: No matching categories. Available: {available}", file=sys.stderr)
            sys.exit(1)

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate adversarial prompts for AI safety red-teaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("red_team_prompts.jsonl"),
        help="Output file path for generated prompts (JSONL format)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("SDG_MODEL", "openai/gpt-4o-mini"),
        help="Teacher model for prompt generation (LiteLLM format, env: SDG_MODEL)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (default: read from environment based on provider)",
    )
    parser.add_argument(
        "--samples-per-concept",
        type=int,
        default=5,
        help="Number of adversarial prompts to generate per harm category",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help="Comma-separated list of categories to include (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the pipeline without making LLM calls",
    )
    parser.add_argument(
        "--flow-id",
        default=None,
        help="SDG Hub flow ID (default: auto-discover red_team/prompt_generation)",
    )

    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    # Parse categories if provided
    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]

    # Build dataset
    print("Building base dataset...")
    base_dataset = build_dataset(categories)
    print(f"  Policy concepts: {len(base_dataset)}")
    print(f"  Categories: {base_dataset['policy_concept'].tolist()}")
    pool_cols = [c for c in base_dataset.columns if c.endswith("_pool")]
    print(f"  Sampling dimensions: {len(pool_cols)} ({', '.join(pool_cols)})")
    print()

    # Discover and load the flow
    print("Loading red_team/prompt_generation flow...")
    FlowRegistry.discover_flows()

    if args.flow_id:
        flow_path = FlowRegistry.get_flow_path(args.flow_id)
    else:
        flow_path = FlowRegistry.get_flow_path("Red Teaming Prompt Generation Flow")

    if flow_path is None:
        print("ERROR: Flow not found in registry.", file=sys.stderr)
        print("Ensure sdg_hub is installed: pip install sdg-hub[examples]", file=sys.stderr)
        sys.exit(1)

    flow = Flow.from_yaml(flow_path)
    print(f"  Flow loaded from: {flow_path}")
    print()

    # Configure the teacher model
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print(
            "Error: No API key found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, "
            "or pass --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.dry_run:
        flow.set_model_config(model=args.model, api_key=api_key)
        print(f"  Model configured: {args.model}")
    print()

    # Generate adversarial prompts
    if args.dry_run:
        print("Dry run — validating pipeline...")
        result = flow.dry_run(base_dataset)
        if isinstance(result, dict):
            final = result.get("final_dataset", {})
            row_count = len(final.get("rows", [])) if isinstance(final, dict) else len(result)
        else:
            row_count = result.shape[0] if hasattr(result, "shape") else len(result)
        print(f"  Validation passed. Expected output: {row_count} rows")
        return

    print(f"Generating adversarial prompts ({args.samples_per_concept} per category)...")
    print(f"  Expected output: ~{len(base_dataset) * args.samples_per_concept} prompts")
    print()

    result = flow.generate(
        base_dataset,
        runtime_params={"replicate_rows": {"num_samples": args.samples_per_concept}},
    )

    print(f"\nGeneration complete: {result.shape[0]} rows, {result.shape[1]} columns")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = result.to_dict(orient="records")
    with open(args.output, "w") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")

    print(f"Results saved to: {args.output}")
    print()

    # Summary
    print("=" * 60)
    print("Generation Summary")
    print("=" * 60)
    if "policy_concept" in result.columns:
        for concept, group in result.groupby("policy_concept"):
            print(f"  {concept}: {len(group)} prompts")
    print(f"\nTotal: {len(result)} adversarial prompts generated")
    print(f"Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
