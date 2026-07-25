# Custom Workbench Images

**Status:** GA

Custom workbench images let you extend the default RHOAI images with additional packages,
frameworks, or internal tools. You can register custom images by creating ImageStream CRDs
on the cluster or by importing them through the RHOAI dashboard. Custom images must follow
the expected base image contract to work correctly as workbenches.

## What's Covered

- Building a custom workbench image from an RHOAI base image
- Dockerfile best practices for adding packages and dependencies
- Registering custom images via ImageStream CRDs
- Importing custom images through the RHOAI dashboard
- Validating that a custom image meets workbench requirements
- Managing image lifecycle and updates

## Official Documentation

- [Provision secure workbenches and custom images for teams](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/provision_secure_workbenches_and_custom_images_for_teams)

## What's in examples/

The `examples/` directory will contain:

- Sample Dockerfiles extending the default JupyterLab and VS Code base images
- ImageStream CR manifests for registering custom images
- A build pipeline example using OpenShift Builds to automate custom image creation
