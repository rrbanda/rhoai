# Workbenches

**Status:** GA

Workbenches are containerized development environments in Red Hat OpenShift AI that provide
ready-to-use IDEs (JupyterLab, VS Code, RStudio) with pre-installed ML frameworks and libraries.
They run as pods on OpenShift, with persistent storage and optional GPU access. Workbenches can be
launched from the RHOAI dashboard or provisioned via CRDs.

## What's Covered

- **IDE Images** -- Default workbench images shipped with RHOAI (JupyterLab, VS Code, RStudio)
  and how to choose the right one for your workload.
- **Custom Images** -- Building and importing custom workbench images with additional packages,
  frameworks, or tools.
- **Data Connections** -- Configuring S3-compatible object storage connections so workbenches can
  read and write data from buckets.

## Directory Layout

```text
02-workbenches/
  ide-images/       # Default IDE image selection and usage
  custom-images/    # Building and registering custom workbench images
  data-connections/  # S3-compatible storage connections
```

## Official Documentation

- [Provision secure workbenches and custom images for teams](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/provision_secure_workbenches_and_custom_images_for_teams)
