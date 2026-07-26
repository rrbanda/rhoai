# Contributing

These guides are community-maintained. Contributions, corrections, and suggestions are welcome.

## Report an Issue

Found an error, broken link, or inaccurate content?

- [Open a GitHub Issue](https://github.com/rrbanda/rhoai/issues/new) with a description of the problem and the page URL

## Suggest a Topic

Have an idea for a new guide or walkthrough?

- [Open a Feature Request](https://github.com/rrbanda/rhoai/issues/new?labels=enhancement) describing the topic and its use case on RHOAI

## Contribute a Guide

1. Fork the [repository](https://github.com/rrbanda/rhoai)
2. Create a branch for your changes
3. Add or edit pages under `docs/` (Markdown with MkDocs Material extensions)
4. Test locally: `mkdocs serve`
5. Submit a pull request

### Writing Guidelines

- Use `mkdocs build --strict` to catch warnings before submitting
- Include `RHOAI Feature:` callouts to link content to specific RHOAI capabilities
- Add `!!! success "Validated on RHOAI X.Y"` admonitions when content has been tested on-cluster
- Use Mermaid diagrams for pipeline flows
- Verify all code snippets against upstream [SDG Hub](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub) and [Training Hub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub) source code

## Related

- [Official RHOAI Documentation](https://docs.redhat.com/en/documentation/red_hat_openshift_ai/latest)
- [SDG Hub Repository](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub)
- [Training Hub Repository](https://github.com/Red-Hat-AI-Innovation-Team/training_hub)
