---
template: home.html
title: Model Customization on RHOAI
hide:
  - toc
---

<div class="home-grid" markdown>

<div class="card" markdown>

### Data Generation

Generate high-quality synthetic training data from your documents using a teacher LLM and SDG Hub's composable flows.

<span class="card-link">[Explore SDG Hub →](data-generation/index.md)</span>

</div>

<div class="card" markdown>

### Model Training

Fine-tune smaller models with SFT, OSFT, LoRA, or GRPO. Preserve base knowledge, minimize GPU usage, or train tool-use agents.

<span class="card-link">[Compare Algorithms →](getting-started/choosing-an-algorithm.md)</span>

</div>

<div class="card" markdown>

### Evaluation

Measure your model's quality with RAG benchmarks, code evaluation, and LLM-as-judge for agent tool-use accuracy.

<span class="card-link">[Evaluate Models →](evaluation/index.md)</span>

</div>

<div class="card" markdown>

### Knowledge Tuning

End-to-end pipeline: prepare documents, generate Q&A data, train with OSFT or SFT, evaluate, and deploy on RHOAI.

<span class="card-link">[Full Walkthrough →](end-to-end/knowledge-tuning.md)</span>

</div>

<div class="card" markdown>

### MCP Distillation

Teach a model to use your tools. A frontier model explores MCP servers and generates training data for GRPO.

<span class="card-link">[Build an Agent →](end-to-end/mcp-distillation.md)</span>

</div>

<div class="card" markdown>

### GPU Planning

Estimate VRAM requirements, compare GPU options, and find tuned hyperparameters for Llama, Qwen, Phi, and Granite.

<span class="card-link">[Plan Resources →](reference/gpu-requirements.md)</span>

</div>

</div>

## Training Algorithms

<div class="algo-badges" markdown>

<a href="training/sft/" class="algo-badge">SFT — Full Fine-Tune</a>
<a href="training/osft/" class="algo-badge">OSFT — Knowledge Preserving</a>
<a href="training/lora/" class="algo-badge">LoRA — Memory Efficient</a>
<a href="training/grpo/" class="algo-badge">GRPO — Tool-Use RL</a>

</div>

## Supported Models

Fine-tuning examples and tuned hyperparameters are provided for:

| Model | Size | Best For |
|-------|------|----------|
| **Llama 3.1** | 8B | General-purpose, strong baseline |
| **Qwen 2.5** | 7B | Multilingual, reasoning |
| **Phi 4 Mini** | 3.8B | Cost-efficient, constrained deployments |
| **Granite 3.3 / 4.0** | 8B | Enterprise workloads |
| **Ministral** | 3B | Ultra-compact environments |
| **GPT-OSS** | 20B | Maximum capacity |

See [Model-Specific Configs](domains/model-specific.md) for per-model hyperparameters and GPU requirements.

## The Pipeline

```mermaid
graph LR
    A[Raw Documents] --> B[SDG Hub]
    B -->|"Q&A pairs, tool traces"| C[Training Hub]
    C -->|"SFT / OSFT / LoRA / GRPO"| D[Fine-Tuned Model]
    D --> E[Evaluation]
    E --> F[KServe / vLLM Serving]
```

**SDG Hub** generates high-quality synthetic training data from your documents using a teacher LLM. **Training Hub** fine-tunes a smaller student model on that data. Both are open-source Python libraries officially referenced in [RHOAI 3.4 documentation](https://docs.redhat.com/en/documentation/red_hat_openshift_ai/3.4) and pre-installed in Red Hat-curated workbench images.
