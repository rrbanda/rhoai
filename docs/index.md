---
template: home.html
title: RHOAI Guides
hide:
  - toc
---

<div class="featured-track" markdown>

### Model Customization

Fine-tune LLMs with smaller, efficient architectures on RHOAI. This track covers the full lifecycle — from generating synthetic training data with **SDG Hub** to training with **Training Hub** (SFT, OSFT, LoRA, GRPO), evaluation, and deployment.

<div class="featured-track-meta" markdown>

**7 training guides** · **3 end-to-end walkthroughs** · **4 algorithms** · **6 supported model families**

</div>

<div class="featured-track-buttons" markdown>

[New to model customization? Start with concepts](getting-started/overview.md){ .md-button .md-button--primary }
[Ready to code? Jump to the quickstart](getting-started/quickstart.md){ .md-button }

</div>

</div>

<div class="track-sections" markdown>

<div class="track-section-item" markdown>

**Knowledge Track** — Teach a model your domain knowledge (financial regulations, medical literature, product docs). Generate Q&A pairs from documents with SDG Hub, then train with SFT or OSFT.
[Knowledge tuning pipeline →](end-to-end/knowledge-tuning.md)

</div>

<div class="track-section-item" markdown>

**Tool-Calling Track** — Fine-tune a model to call tools from MCP servers, APIs, and databases. Generate expert tool-use traces with MCP distillation, then train with LoRA SFT. Validated end-to-end on RHOAI 3.4.2.
[Tool-calling model pipeline (financial example) →](end-to-end/tool-calling-financial.md)

</div>

<div class="track-section-item" markdown>

**Training Algorithms** — Compare SFT, OSFT, LoRA, and GRPO side by side and pick the right one for your constraints.
[Choose an algorithm →](getting-started/choosing-an-algorithm.md)

</div>

<div class="track-section-item" markdown>

**Serving & Guardrails** — Deploy fine-tuned models with KServe + vLLM and protect them with NeMo Guardrails.
[Deploy →](serving/index.md) · [Guardrails →](guardrails/index.md)

</div>

<div class="track-section-item" markdown>

**Evaluation & Utilities** — Measure quality with RAG benchmarks, estimate GPU memory, visualize training loss.
[Evaluation →](evaluation/index.md) · [GPU planning →](reference/gpu-requirements.md)

</div>

</div>

!!! note "Disclaimer"
    These guides are community-maintained and intended as supplementary learning material. For official product documentation, refer to the [latest Red Hat OpenShift AI documentation](https://docs.redhat.com/en/documentation/red_hat_openshift_ai/latest).
