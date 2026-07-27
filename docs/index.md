---
template: home.html
title: RHOAI Guides
hide:
  - toc
---

<div class="featured-track" markdown>

### :material-tune-vertical: Model Customization on RHOAI

End-to-end practical guides for fine-tuning LLMs on Red Hat OpenShift AI — from raw documents to a deployed, guardrailed model. These guides build on the [official RHOAI platform docs](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4) with integrated workflows using **SDG Hub** (data generation) and **Training Hub** (SFT, OSFT, LoRA, GRPO).

<div class="featured-track-meta" markdown>

:material-check-decagram: **Validated on RHOAI 3.4.2** · 7 training guides · 3 end-to-end pipelines · 4 algorithms · 6 model families

</div>

<div class="featured-track-buttons" markdown>

[Start with concepts](getting-started/overview.md){ .md-button .md-button--primary }
[Quickstart](getting-started/quickstart.md){ .md-button }
[Choose an algorithm](getting-started/choosing-an-algorithm.md){ .md-button }

</div>

</div>

<div class="track-sections" markdown>

<div class="track-section-item" markdown>

:material-book-open-variant: **Knowledge Tuning** — Teach a model your domain knowledge from documents. Generate Q&A data with SDG Hub, then train with SFT, OSFT, or LoRA.
[Knowledge tuning pipeline :material-arrow-right:](end-to-end/knowledge-tuning.md)

</div>

<div class="track-section-item" markdown>

:material-wrench: **Tool-Calling Model** — Fine-tune a model to call tools from MCP servers. Generate expert traces with MCP distillation, then train with LoRA SFT.
[Tool-calling pipeline :material-arrow-right:](end-to-end/tool-calling-financial.md)

</div>

<div class="track-section-item" markdown>

:material-scale-balance: **Training Algorithms** — Compare SFT, OSFT, LoRA, and GRPO side by side. Pick the right one for your GPU budget and use case.
[Choose an algorithm :material-arrow-right:](getting-started/choosing-an-algorithm.md)

</div>

<div class="track-section-item" markdown>

:material-database-arrow-right: **Data Generation** — Use SDG Hub to create high-quality synthetic training data from documents, knowledge bases, or MCP servers.
[SDG Hub overview :material-arrow-right:](data-generation/index.md)

</div>

<div class="track-section-item" markdown>

:material-rocket-launch: **Deploy & Serve** — Deploy fine-tuned models on RHOAI with KServe + vLLM. Add NeMo Guardrails for safety.
[Serving :material-arrow-right:](serving/index.md) · [Guardrails :material-arrow-right:](guardrails/index.md)

</div>

<div class="track-section-item" markdown>

:material-chart-bar: **Evaluate & Measure** — RAG benchmarks, tool-use scoring, code evaluation, GPU memory estimation, loss visualization.
[Evaluation :material-arrow-right:](evaluation/index.md) · [GPU planning :material-arrow-right:](reference/gpu-requirements.md)

</div>

</div>

<div class="site-disclaimer" markdown>

These guides are community-maintained supplementary material. For official product documentation, see the [Red Hat OpenShift AI docs](https://docs.redhat.com/en/documentation/red_hat_openshift_ai/latest).

</div>
