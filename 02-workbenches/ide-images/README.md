# IDE Images

**Status:** GA

Red Hat OpenShift AI ships default workbench images for JupyterLab, VS Code, and RStudio.
Each image comes with pre-installed ML libraries such as TensorFlow, PyTorch, and common
data science tooling. Choosing the right image depends on your preferred IDE and the
frameworks your workload requires.

## What's Covered

- Available IDE images and the libraries bundled with each
- Differences between minimal, standard, and CUDA-enabled image variants
- Selecting the right image based on framework needs (TensorFlow, PyTorch, etc.)
- GPU-accelerated images and when to use them
- Image versioning and update cadence

## Official Documentation

- [Use the Red Hat data science IDE images effectively](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/use_the_red_hat_data_science_ide_images_effectively)

## What's in examples/

The `examples/` directory will contain:

- Sample notebooks demonstrating how to verify installed libraries and GPU availability
- Scripts for listing available image tags and comparing image contents
- Workbench CR manifests showing how to launch specific IDE images via CRDs
