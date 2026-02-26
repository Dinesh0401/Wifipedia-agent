# Wifipediaagent

MIPROv2 + ACE Research Pipeline for HotpotQA with LLMControls API.

## Overview

A Wikipedia-based agent benchmark system that uses advanced retrieval and reasoning pipelines to answer multi-hop questions from the HotpotQA dataset.

## Features

- **Wiki Retriever**: Wikipedia-based document retrieval for question answering
- **Wiki Agent**: LLM-powered agent for multi-hop reasoning over Wikipedia articles
- **MIPROv2 Pipeline**: Automated prompt optimization pipeline
- **ACE Pipeline**: Adaptive computation engine for efficient inference
- **HotpotQA Loader**: Dataset loader for the HotpotQA benchmark
- **LLMControls Client**: API client for LLM orchestration
- **Metrics**: Evaluation metrics for benchmark scoring

## Project Structure

```
Wifipediaagent/
├── benchmarks/
│   └── tasks/
│       └── hotpotqa.yaml
├── scripts/
│   ├── ace_pipeline.py
│   ├── config.py
│   ├── hotpotqa_loader.py
│   ├── llmcontrols_client.py
│   ├── metrics.py
│   ├── miprov2_pipeline.py
│   ├── run_pipeline.py
│   ├── wiki_agent.py
│   └── wiki_retriever.py
├── optimized_programs/
├── research_outputs/
├── pyproject.toml
└── README.md
```

## Installation

```bash
pip install -e .
```

## Usage

```bash
run-pipeline
```

Or directly:

```bash
python -m scripts.run_pipeline
```

## Requirements

- Python >= 3.10
- See `pyproject.toml` for full dependency list
