---
template: home.html
title: RHOAI Guides
hide:
  - toc
---

<div class="featured-track" markdown>

### :material-tune-vertical: Model Customization on RHOAI

End-to-end practical guides for fine-tuning LLMs on Red Hat OpenShift AI — from raw documents to a deployed, guardrailed model that can power an autonomous agent. These guides build on the [official RHOAI platform docs](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4) with integrated workflows using **SDG Hub** (data generation) and **Training Hub** (SFT, OSFT, LoRA, GRPO).

<div class="featured-track-meta" markdown>

:material-check-decagram: **Validated on RHOAI 3.4.2** · 7 training guides · 4 end-to-end pipelines · 4 algorithms · 6 model families

</div>

<div class="featured-track-buttons" markdown>

[Start here: Concepts](getting-started/overview.md){ .md-button .md-button--primary }
[Environment setup](getting-started/setup.md){ .md-button }

</div>

</div>

<div class="section-heading" markdown>

#### Pipelines — pick one and follow it end to end

</div>

<div class="track-sections" markdown>

<div class="track-section-item" markdown>

:material-book-open-variant: **Knowledge Tuning**

Teach a model your domain knowledge from documents. Generate Q&A data with SDG Hub, then train with SFT, OSFT, or LoRA.

[Follow the pipeline :material-arrow-right:](end-to-end/knowledge-tuning.md)

</div>

<div class="track-section-item" markdown>

:material-wrench: **Tool-Calling Model + Deep Agent**

Fine-tune a model to call tools from MCP servers, then wire it into an autonomous agent with planning, memory, and skills.

[Train the model :material-arrow-right:](end-to-end/tool-calling-financial.md) · [Build the agent :material-arrow-right:](end-to-end/deep-agent.md)

</div>

<div class="track-section-item" markdown>

:material-flask-outline: **MCP Distillation (GRPO)**

Train with reinforcement learning instead of SFT. Same MCP distillation data, different training algorithm.

[Follow the pipeline :material-arrow-right:](end-to-end/mcp-distillation.md)

</div>

</div>

<div class="section-heading" markdown>

#### Reference — dive deeper into individual topics

</div>

<div class="track-sections" markdown>

<div class="track-section-item" markdown>

:material-scale-balance: **Choosing an Algorithm** — Compare SFT, OSFT, LoRA, and GRPO side by side.
[Guide :material-arrow-right:](getting-started/choosing-an-algorithm.md)

</div>

<div class="track-section-item" markdown>

:material-database-arrow-right: **Data Generation** — Use SDG Hub for synthetic training data.
[SDG Hub overview :material-arrow-right:](data-generation/index.md)

</div>

<div class="track-section-item" markdown>

:material-rocket-launch: **Deploy & Serve** — KServe + vLLM with NeMo Guardrails.
[Serving :material-arrow-right:](serving/index.md) · [Guardrails :material-arrow-right:](guardrails/index.md)

</div>

<div class="track-section-item" markdown>

:material-chart-bar: **Evaluate** — RAG benchmarks, tool-use scoring, GPU estimation.
[Evaluation :material-arrow-right:](evaluation/index.md) · [GPU planning :material-arrow-right:](reference/gpu-requirements.md)

</div>

</div>

<div class="site-disclaimer" markdown>

These guides are community-maintained supplementary material. For official product documentation, see the [Red Hat OpenShift AI docs](https://docs.redhat.com/en/documentation/red_hat_openshift_ai/latest).

</div>
