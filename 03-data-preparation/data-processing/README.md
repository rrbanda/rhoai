# Data Processing with Docling

**Status:** GA (General Availability)

Data processing covers document ingestion and preprocessing using Docling. It converts URLs, PDFs, and other document formats into structured Markdown, then chunks the text and produces seed datasets suitable for synthetic data generation with SDG Hub.

## What's Covered

- Ingesting documents from URLs, PDFs, and other supported formats
- Converting documents to structured Markdown with Docling
- Chunking text into appropriately sized segments
- Producing seed datasets (taxonomy files) for SDG Hub flows
- Configuring Docling options for different document types

## Official Documentation

- [Customize Models for Gen AI and Agentic AI](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/customize_models_for_gen_ai_and_agentic_ai_applications)

## What's in examples/

Examples demonstrate end-to-end document processing workflows -- from ingesting raw documents (PDFs, web pages) through Markdown conversion and chunking to producing seed datasets ready for SDG pipelines.
