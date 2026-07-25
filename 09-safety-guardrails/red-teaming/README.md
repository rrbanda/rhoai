# Red Teaming — Adversarial Prompt Generation

## Overview

Red teaming is a critical component of AI safety evaluation. This section demonstrates how to use **SDG Hub** to systematically generate diverse adversarial prompts for testing LLM safety guardrails.

The `red_team/prompt_generation` flow creates adversarial prompts by combining multi-dimensional sampling (demographics, expertise, geography, language styles, exploit stages, etc.) with LLM-powered generation. This produces realistic, varied attack scenarios that stress-test content policies across multiple harm categories.

## How It Works

The flow pipeline operates in stages:

1. **Replicate** — Each input row (policy concept) is replicated N times to produce multiple samples
2. **Multi-dimensional sampling** — For each replica, the flow samples from configurable pools (demographics, expertise, geography, language style, exploit stage, task medium, temporal context, trust signals)
3. **Prompt construction** — Sampled dimensions are combined into a structured prompt template
4. **LLM generation** — A teacher model generates adversarial prompts based on the constructed context
5. **Response parsing** — JSON responses are parsed to extract the generated prompt and reasoning

## Harm Categories

The default taxonomy includes:

| Category | Description |
|----------|-------------|
| Illegal Activity | Eliciting advice for carrying out illegal acts |
| Hate Speech | Generating discriminatory or bullying content |
| Security & Malware | Exploiting systems or creating malware |
| Violence | Content related to physical harm |
| Fraud | Strategies for committing fraud |
| Sexually Explicit | Generating explicit sexual content |
| Misinformation | Creating or promoting false information |
| Self Harm | Advice on inflicting self-harm |

## Prerequisites

- Python 3.10+
- `sdg_hub` installed (`pip install sdg_hub`)
- An LLM API key (OpenAI, Anthropic, or any LiteLLM-compatible provider)

## What's in examples/

| File | Description |
|------|-------------|
| [`generate_red_team_prompts.py`](examples/generate_red_team_prompts.py) | Standalone script that generates adversarial prompts across all harm categories |

### Quick Start

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Generate adversarial prompts (default: 5 samples per category = 40 total)
python examples/generate_red_team_prompts.py --output red_team_prompts.jsonl

# Use a different model
python examples/generate_red_team_prompts.py \
    --model anthropic/claude-sonnet-4-20250514 \
    --samples-per-concept 10

# Dry run (validate pipeline without LLM calls)
python examples/generate_red_team_prompts.py --dry-run
```

## Integration with Guardrails

The generated adversarial prompts can be used to:

1. **Test NeMo Guardrails** configurations (see `../nemo-guardrails/`)
2. **Evaluate content filters** before deploying to production
3. **Benchmark safety classifiers** with diverse, realistic attack vectors
4. **Red-team custom models** fine-tuned on RHOAI

## References

- [SDG Hub Documentation](https://github.com/redhat-ai/sdg_hub)
- [Red Team Prompt Generation Flow](https://github.com/redhat-ai/sdg_hub/tree/main/src/sdg_hub/flows/red_team/prompt_generation)
- [RHOAI Safety & Guardrails](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/ensuring_ai_safety_with_guardrails)
