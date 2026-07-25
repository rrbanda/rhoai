# LM-Eval

**Status:** GA

LM-Eval is the evaluation framework in RHOAI 3.4 for assessing language model performance against standardized benchmarks. It uses the LMEvalJob custom resource to define evaluation runs, select benchmark tasks, and produce metrics that enable objective model comparison.

## What's Covered

- Creating and configuring LMEvalJob CRDs
- Selecting benchmark tasks (e.g., MMLU, HellaSwag, ARC)
- Running evaluations against served or local models
- Retrieving metrics and interpreting results
- Comparing performance across model versions or configurations

## Official Documentation

- [Evaluating AI Systems](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/evaluating_ai_systems)

## What's in examples/

Examples will include LMEvalJob YAML manifests for common benchmarks, scripts to trigger evaluations programmatically, and notebooks for visualizing evaluation results.
