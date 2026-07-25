# Backup and Restore

**Status:** GA

Data backup and restore procedures for RHOAI 3.4 platform data. Protect platform configuration, pipeline definitions, model metadata, and user workspaces against data loss with systematic backup strategies.

## What's Covered

- Identifying critical platform data for backup
- Backing up RHOAI operator configuration and CRDs
- Backing up pipeline definitions and run history
- Restoring platform state from backups
- Scheduling automated backup jobs

## Official Documentation

- [Configure User Access, Storage, and Telemetry in OpenShift AI](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/configure_user_access_storage_and_telemetry_in_openshift_ai)

## What's in examples/

Examples will include backup scripts for key resources, CronJob manifests for scheduled backups, restore procedures, and validation scripts to verify backup integrity.
