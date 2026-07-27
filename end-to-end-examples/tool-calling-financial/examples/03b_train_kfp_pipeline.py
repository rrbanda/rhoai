"""Run the LoRA fine-tuning KFP pipeline on RHOAI.

This script compiles and uploads the official LoRA training pipeline from
red-hat-data-services/pipelines-components, then creates a pipeline run
with parameters configured for the tool-calling financial model use case.

The pipeline executes four stages on the RHOAI cluster:
  1. Dataset Download — fetches training data from HuggingFace or S3
  2. LoRA Training — fine-tunes the base model with LoRA adapters via
     Kubeflow Trainer + Training Hub's Unsloth backend
  3. Evaluation — runs LM-Eval harness benchmarks on the fine-tuned model
  4. Model Registry — registers the model to Kubeflow Model Registry (optional)

Prerequisites:
  - RHOAI 3.4+ with dashboard, trainer, and aipipelines components enabled
  - Pipeline server running in your Data Science Project
  - Storage class available (default: gp3-csi with ReadWriteOnce)
  - kubernetes-credentials secret created (see README.md)

Source:
  Pipeline:  https://github.com/red-hat-data-services/pipelines-components/tree/main/pipelines/training/finetuning/lora
  Component: https://github.com/red-hat-data-services/pipelines-components/tree/main/components/training/finetuning/lora
  Guide:     https://github.com/red-hat-data-services/red-hat-ai-examples/tree/main/examples/fine-tuning/pipelines/training-hub

Usage:
    # Generate pipeline YAML (requires pipelines-components clone):
    python 03b_train_kfp_pipeline.py --compile-only

    # Upload and run (requires RHOAI cluster access):
    python 03b_train_kfp_pipeline.py \\
        --dataset-uri hf://LipengCS/Table-GPT:All \\
        --base-model Qwen/Qwen3-4B \\
        --namespace tool-calling-financial

    # With local SDG data pushed to S3:
    python 03b_train_kfp_pipeline.py \\
        --dataset-uri s3://my-bucket/tool-calling-financial/training_data.jsonl \\
        --base-model Qwen/Qwen3-4B
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


PIPELINES_REPO = "https://github.com/red-hat-data-services/pipelines-components.git"
PIPELINES_DIR = "pipelines-components"
PIPELINE_SCRIPT = "pipelines/training/finetuning/lora/pipeline.py"
PIPELINE_YAML = "pipelines/training/finetuning/lora/pipeline.yaml"
KFP_VERSION = "2.15.2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile and run the LoRA KFP pipeline on RHOAI"
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only compile the pipeline YAML, do not upload or run",
    )
    parser.add_argument(
        "--pipeline-yaml",
        type=str,
        default=None,
        help="Path to pre-compiled pipeline YAML (skip clone+compile)",
    )
    parser.add_argument(
        "--dataset-uri",
        type=str,
        default="hf://LipengCS/Table-GPT:All",
        help="Dataset URI: hf://dataset, s3://bucket/path, or https://url "
        "(default: hf://LipengCS/Table-GPT:All)",
    )
    parser.add_argument(
        "--dataset-subset",
        type=int,
        default=2000,
        help="Limit to first N examples; 0 = all (default: 2000)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model HuggingFace ID (default: STUDENT_MODEL env or Qwen/Qwen3-4B)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA scaling factor (default: 32)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=2,
        help="Training epochs (default: 2)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--eval-tasks",
        type=str,
        nargs="+",
        default=["arc_easy"],
        help="LM-Eval tasks (default: arc_easy)",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="RHOAI namespace (default: OPENSHIFT_NAMESPACE env or tool-calling-financial)",
    )
    parser.add_argument(
        "--registry-address",
        type=str,
        default="",
        help="Model Registry address; empty = skip registration (default: empty)",
    )
    parser.add_argument(
        "--storage-class",
        type=str,
        default="gp3-csi",
        help="Storage class name (default: gp3-csi with ReadWriteOnce)",
    )
    return parser.parse_args()


def clone_and_compile(storage_class: str) -> Path:
    """Clone pipelines-components repo and compile the LoRA pipeline YAML."""
    if not Path(PIPELINES_DIR).exists():
        print(f"Cloning {PIPELINES_REPO}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", PIPELINES_REPO, PIPELINES_DIR],
            check=True,
        )
    else:
        print(f"Using existing clone at {PIPELINES_DIR}/")

    pipeline_script = Path(PIPELINES_DIR) / PIPELINE_SCRIPT
    if not pipeline_script.exists():
        print(f"ERROR: Pipeline script not found at {pipeline_script}")
        print("The pipelines-components repository structure may have changed.")
        sys.exit(1)

    if storage_class != "gp3-csi":
        print(f"Updating PVC_STORAGE_CLASS to '{storage_class}'...")
        content = pipeline_script.read_text()
        content = content.replace(
            'PVC_STORAGE_CLASS = "gp3-csi"',
            f'PVC_STORAGE_CLASS = "{storage_class}"',
        )
        pipeline_script.write_text(content)

    print(f"Compiling pipeline (kfp=={KFP_VERSION})...")
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q",
            f"kfp=={KFP_VERSION}", f"kfp-kubernetes=={KFP_VERSION}",
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(pipeline_script)],
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(PIPELINES_DIR).resolve())},
    )

    yaml_path = Path(PIPELINES_DIR) / PIPELINE_YAML
    if not yaml_path.exists():
        print(f"ERROR: Compiled YAML not found at {yaml_path}")
        sys.exit(1)

    print(f"Pipeline compiled: {yaml_path}")
    return yaml_path


def upload_and_run(yaml_path: Path, args: argparse.Namespace) -> None:
    """Upload pipeline to RHOAI and create a run."""
    try:
        import kfp
    except ImportError:
        print("ERROR: kfp package required. Install with: pip install kfp==2.15.2")
        sys.exit(1)

    namespace = args.namespace or os.environ.get("OPENSHIFT_NAMESPACE", "tool-calling-financial")
    base_model = args.base_model or os.environ.get("STUDENT_MODEL", "Qwen/Qwen3-4B")

    kfp_endpoint = os.environ.get("KFP_ENDPOINT")
    if not kfp_endpoint:
        print("ERROR: KFP_ENDPOINT env var required (e.g., https://ds-pipeline-dspa.apps.cluster.com)")
        print("Find it in RHOAI Dashboard > Data Science Pipelines > Pipeline Server details")
        sys.exit(1)

    print(f"\nConnecting to KFP at {kfp_endpoint}...")
    kfp_client = kfp.Client(host=kfp_endpoint)

    print(f"Uploading pipeline from {yaml_path}...")
    pipeline = kfp_client.upload_pipeline(
        pipeline_package_path=str(yaml_path),
        pipeline_name="tool-calling-financial-lora-training",
        description="LoRA fine-tuning pipeline for the tool-calling financial model",
    )

    params = {
        "phase_01_dataset_man_data_uri": args.dataset_uri,
        "phase_01_dataset_opt_subset": args.dataset_subset,
        "phase_02_train_man_train_model": base_model,
        "phase_02_train_man_train_epochs": args.num_epochs,
        "phase_02_train_man_lora_r": args.lora_r,
        "phase_02_train_man_lora_alpha": args.lora_alpha,
        "phase_02_train_opt_learning_rate": args.learning_rate,
        "phase_02_train_opt_lora_load_in_4bit": True,
        "phase_02_train_opt_use_liger": True,
        "phase_02_train_opt_lr_scheduler": "cosine",
        "phase_03_eval_man_eval_tasks": args.eval_tasks,
        "phase_04_registry_man_address": args.registry_address,
        "phase_04_registry_man_reg_name": "tool-calling-financial-lora",
    }

    print(f"\nCreating pipeline run...")
    print(f"  Dataset:    {args.dataset_uri}")
    print(f"  Model:      {base_model}")
    print(f"  LoRA r:     {args.lora_r}")
    print(f"  Epochs:     {args.num_epochs}")
    print(f"  Eval tasks: {args.eval_tasks}")

    run = kfp_client.create_run_from_pipeline_package(
        pipeline_file=str(yaml_path),
        arguments=params,
        run_name=f"tool-calling-financial-lora-{base_model.split('/')[-1]}",
        experiment_name="tool-calling-financial",
        namespace=namespace,
    )

    print(f"\nPipeline run created!")
    print(f"  Run ID: {run.run_id}")
    print(f"  Monitor in RHOAI Dashboard > Pipelines > Runs")
    print(f"\nThe pipeline will:")
    print(f"  1. Download and validate the dataset")
    print(f"  2. Fine-tune {base_model} with LoRA (rank={args.lora_r})")
    print(f"  3. Evaluate with LM-Eval ({', '.join(args.eval_tasks)})")
    if args.registry_address:
        print(f"  4. Register model to {args.registry_address}")
    else:
        print(f"  4. Skip model registry (no address provided)")


def main() -> None:
    load_dotenv()
    args = parse_args()

    print("=" * 60)
    print("tool-calling financial model — LoRA KFP Pipeline on RHOAI")
    print("=" * 60)

    if args.pipeline_yaml:
        yaml_path = Path(args.pipeline_yaml)
        if not yaml_path.exists():
            print(f"ERROR: Pipeline YAML not found at {yaml_path}")
            sys.exit(1)
    else:
        yaml_path = clone_and_compile(args.storage_class)

    if args.compile_only:
        print(f"\nCompile-only mode. Pipeline YAML: {yaml_path}")
        print("Upload this file to RHOAI Dashboard > Pipelines > Import Pipeline")
        return

    upload_and_run(yaml_path, args)


if __name__ == "__main__":
    main()
