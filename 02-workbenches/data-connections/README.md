# Data Connections

**Status:** GA

Data connections provide workbenches with access to S3-compatible object storage (such as
AWS S3, MinIO, or Ceph). Once configured, connection credentials are injected as environment
variables, allowing boto3 and other S3 clients to authenticate automatically. Connections can
be created through the RHOAI dashboard or by applying Secret-based CRDs.

## What's Covered

- Creating data connections via the RHOAI dashboard
- Defining data connections as Kubernetes Secrets / CRDs
- Environment variables injected into the workbench pod
- Using boto3 to list, read, and write objects in connected buckets
- Sharing connections across multiple workbenches
- Troubleshooting common connectivity and permission issues

## Official Documentation

- [Connect your workbench to S3-compatible object storage](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/connect_your_workbench_to_s3-compatible_object_storage)

## What's in examples/

The `examples/` directory will contain:

- Secret manifests for creating data connections via CRDs
- Python scripts demonstrating boto3 usage (listing buckets, uploading/downloading files)
- A notebook showing how to load datasets directly from S3 into pandas DataFrames
