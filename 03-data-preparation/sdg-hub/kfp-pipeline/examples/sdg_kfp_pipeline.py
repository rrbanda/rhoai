"""SDG Hub Kubeflow Pipeline for Knowledge Data Generation.

Defines a three-stage KFP v2 pipeline that generates synthetic training
data using SDG Hub flows, validates the output, and uploads results to
S3 or a PVC for downstream training.  Each stage is a lightweight
container component with explicit inputs and outputs.

Usage:
    # Compile the pipeline to YAML
    python sdg_kfp_pipeline.py

    # Compile with a custom output path
    python sdg_kfp_pipeline.py --output sdg_pipeline.yaml
"""

from __future__ import annotations

import argparse

from kfp import compiler, dsl


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-generic-data-science-notebook-rhel9:3.4",
    packages_to_install=["sdg_hub[examples]", "pandas", "python-dotenv"],
)
def generate_data(
    model_endpoint: str,
    model_api_key: str,
    flow_variant: str,
    document_jsonl: dsl.Input[dsl.Dataset],
    generated_data: dsl.Output[dsl.Dataset],
    checkpoint_dir: str = "/tmp/sdg_checkpoints",
) -> None:
    """Run an SDG Hub knowledge flow to produce JSONL training data.

    Reads pre-processed documents from *document_jsonl*, configures the
    requested *flow_variant*, and writes the generated Q&A pairs to the
    *generated_data* output artifact.
    """
    import os

    import pandas as pd

    from sdg_hub import Flow, FlowRegistry

    flow_name_map = {
        "extractive_summary": "Extractive Summary Knowledge Tuning Dataset Generation Flow",
        "detailed_summary": "Detailed Summary Knowledge Tuning Dataset Generation Flow",
        "key_facts": "Key Facts Knowledge Tuning Dataset Generation Flow",
        "doc_direct_qa": "Document Based Knowledge Tuning Dataset Generation Flow",
    }

    flow_display = flow_name_map.get(flow_variant)
    if flow_display is None:
        raise ValueError(
            f"Unknown flow variant '{flow_variant}'. "
            f"Choose from: {list(flow_name_map.keys())}"
        )

    FlowRegistry.discover_flows()
    flow_path = FlowRegistry.get_flow_path(flow_display)
    if flow_path is None:
        raise RuntimeError(f"Flow '{flow_display}' not found in SDG Hub registry.")

    dataset = pd.read_json(document_jsonl.path, lines=True)
    print(f"Loaded {len(dataset)} documents from input artifact")

    flow = Flow.from_yaml(flow_path)

    api_base = model_endpoint.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base += "/v1"

    flow.set_model_config(
        model=f"openai/{model_endpoint.split('/')[-1]}",
        api_key=model_api_key,
        api_base=api_base,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Running SDG Hub flow: {flow_variant}")
    result = flow.generate(dataset, checkpoint_dir=checkpoint_dir)

    if isinstance(result, pd.DataFrame):
        result_df = result
    else:
        result_df = result.to_pandas()

    result_df.to_json(generated_data.path, orient="records", lines=True)
    print(f"Generated {len(result_df)} Q&A pairs -> {generated_data.path}")


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-generic-data-science-notebook-rhel9:3.4",
    packages_to_install=["pandas"],
)
def validate_data(
    generated_data: dsl.Input[dsl.Dataset],
    validated_data: dsl.Output[dsl.Dataset],
    min_rows: int = 1,
    required_columns: str = "question,response",
) -> None:
    """Validate the generated JSONL for schema and quality.

    Checks that the dataset is non-empty, contains the required columns,
    and that no rows have blank question/response fields.  Copies valid
    rows to *validated_data* and raises on fatal issues.
    """
    import pandas as pd

    df = pd.read_json(generated_data.path, lines=True)
    print(f"Loaded {len(df)} rows for validation")

    columns = [c.strip() for c in required_columns.split(",")]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    initial_count = len(df)

    df = df.dropna(subset=columns)
    for col in columns:
        df = df[df[col].astype(str).str.strip().str.len() > 0]

    dropped = initial_count - len(df)
    if dropped > 0:
        print(f"WARNING: Dropped {dropped} rows with empty fields")

    if len(df) < min_rows:
        raise ValueError(
            f"Validation failed: only {len(df)} valid rows "
            f"(minimum required: {min_rows})"
        )

    df.to_json(validated_data.path, orient="records", lines=True)
    print(f"Validation passed: {len(df)} valid rows")
    print(f"  Columns   : {list(df.columns)}")
    print(f"  Row count : {len(df)}")


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-generic-data-science-notebook-rhel9:3.4",
    packages_to_install=["boto3", "pandas"],
)
def upload_data(
    validated_data: dsl.Input[dsl.Dataset],
    output_path: str,
    s3_endpoint: str = "",
    s3_bucket: str = "",
    s3_access_key: str = "",
    s3_secret_key: str = "",
) -> None:
    """Upload validated data to S3 or a local PVC path.

    When *s3_bucket* is provided, uploads to the given S3 bucket via
    *s3_endpoint*.  Otherwise, copies the data to *output_path* on a
    shared PVC.
    """
    import shutil

    import pandas as pd

    df = pd.read_json(validated_data.path, lines=True)
    print(f"Uploading {len(df)} rows")

    if s3_bucket:
        import boto3

        client_kwargs = {}
        if s3_endpoint:
            client_kwargs["endpoint_url"] = s3_endpoint

        s3 = boto3.client(
            "s3",
            aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key,
            **client_kwargs,
        )

        s3_key = output_path.lstrip("/")
        s3.upload_file(validated_data.path, s3_bucket, s3_key)
        print(f"Uploaded to s3://{s3_bucket}/{s3_key}")
    else:
        import os

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        shutil.copy2(validated_data.path, output_path)
        print(f"Copied to PVC path: {output_path}")


@dsl.pipeline(
    name="SDG Hub Knowledge Data Generation",
    description=(
        "Three-stage pipeline: generate synthetic Q&A data with SDG Hub, "
        "validate output quality, and upload to S3 or PVC for training."
    ),
)
def sdg_knowledge_pipeline(
    model_endpoint: str = "https://your-model-endpoint.example.com",
    model_api_key: str = "your-api-key",
    flow_variant: str = "extractive_summary",
    document_jsonl: str = "s3://my-bucket/documents/input.jsonl",
    output_path: str = "/data/sdg-output/generated_data.jsonl",
    min_rows: int = 10,
    s3_endpoint: str = "",
    s3_bucket: str = "",
    s3_access_key: str = "",
    s3_secret_key: str = "",
) -> None:
    """Orchestrate SDG data generation, validation, and upload."""
    doc_artifact = dsl.importer(
        artifact_uri=document_jsonl,
        artifact_class=dsl.Dataset,
        reimport=True,
    )

    generate_task = generate_data(
        model_endpoint=model_endpoint,
        model_api_key=model_api_key,
        flow_variant=flow_variant,
        document_jsonl=doc_artifact.output,
    )
    generate_task.set_display_name("Generate Synthetic Data")

    validate_task = validate_data(
        generated_data=generate_task.outputs["generated_data"],
        min_rows=min_rows,
    )
    validate_task.set_display_name("Validate Output")

    upload_task = upload_data(
        validated_data=validate_task.outputs["validated_data"],
        output_path=output_path,
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
    )
    upload_task.set_display_name("Upload to Storage")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile the SDG Hub KFP pipeline to YAML."
    )
    parser.add_argument(
        "--output",
        default="sdg_knowledge_pipeline.yaml",
        help="Output YAML file path (default: sdg_knowledge_pipeline.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compiler.Compiler().compile(
        pipeline_func=sdg_knowledge_pipeline,
        package_path=args.output,
    )
    print(f"Pipeline compiled to {args.output}")


if __name__ == "__main__":
    main()
