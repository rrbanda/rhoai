# AutoML

**Status:** DP (Developer Preview)

AutoML automates model selection and hyperparameter tuning, reducing the manual effort required to find optimal training configurations. It systematically explores algorithm and parameter combinations to identify the best-performing model for a given dataset and objective.

## What's Covered

- Automated model selection across supported algorithms
- Hyperparameter search and optimization
- Evaluation metric tracking and comparison
- Integration with RHOAI training infrastructure

## Key Concepts

- **Search space** -- The set of algorithms and hyperparameter ranges to explore
- **Objective metric** -- The evaluation criterion used to rank candidate models
- **Trial budget** -- Maximum number of configurations to evaluate

## Official Documentation

- [Working with AutoML](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/working_with_automl)

## What's in examples/

Examples show how to define search spaces, configure optimization objectives, launch AutoML experiments, and compare trial results to select the best model.
