<p align="center">
  <h1 align="center">WikiQA-Bench</h1>
  <p align="center">
    <strong>MIPROv2 + ACE Research Pipeline for HotpotQA with LLM-as-a-Judge Evaluation</strong>
  </p>
  <p align="center">
    <a href="#quickstart">Quickstart</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#results">Results</a> •
    <a href="#configuration">Configuration</a> •
    <a href="#contributing">Contributing</a>
  </p>
</p>

---

## Table of Contents

- [About](#about)
- [Architecture](#architecture)
- [Pipeline Stages](#pipeline-stages)
- [Results](#results)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Environment Variables](#environment-variables)
- [Key Components](#key-components)
- [Evaluation Methodology](#evaluation-methodology)
- [Contributing](#contributing)
- [License](#license)

---

## About

**WikiQA-Bench** is a research-grade benchmark system that evaluates LLM-powered Wikipedia retrieval agents on the [HotpotQA](https://hotpotqa.github.io/) dataset. It implements a three-stage evaluation pipeline:

1. **MIPROv2** — Automated prompt optimization via [DSPy](https://github.com/stanfordnlp/dspy)
2. **ACE Offline** — Single-pass faithfulness and accuracy evaluation
3. **ACE Online** — Iterative self-reflection with an adaptive skillbook

All stages use **LLM-as-a-Judge** binary evaluation with a separate judge endpoint to avoid self-evaluation bias.

### Key Highlights

- **Multi-backend LLM support** — LLMControls, Anthropic Claude, and DeepSeek (via OpenRouter)
- **Automated prompt optimization** — MIPROv2 discovers high-quality chain-of-thought prompts
- **Faithfulness auditing** — ACE verifies if answers are grounded in retrieved context
- **Confidence intervals** — Wilson score 95% CI on all accuracy metrics
- **JSONL provenance** — Full per-question predictions, retrieval context, and judge reasoning saved for reproducibility

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        run_pipeline.py                          │
│                     (CLI Orchestrator)                           │
├──────────┬──────────────────┬───────────────────────────────────┤
│          │                  │                                   │
│  Stage 1 │      Stage 2     │           Stage 3                 │
│ MIPROv2  │   ACE Offline    │         ACE Online                │
│          │                  │                                   │
│ ┌──────┐ │ ┌──────────────┐ │ ┌──────────────────────────────┐  │
│ │DSPy  │ │ │Single-pass   │ │ │Iterative adaptation with     │  │
│ │Prompt│ │ │faithfulness  │ │ │self-reflection & skillbook   │  │
│ │Optim.│ │ │+ accuracy    │ │ │(max_reflections per sample)  │  │
│ └──┬───┘ │ └──────┬───────┘ │ └──────────────┬───────────────┘  │
│    │     │        │         │                │                  │
├────┴─────┴────────┴─────────┴────────────────┴──────────────────┤
│                                                                 │
│   WikiAgent  ←→  WikipediaRetriever  ←→  UnifiedLLMClient      │
│                                                                 │
│   LLMJudge (separate endpoint, temperature=0.0)                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages

### Stage 1 — MIPROv2 (Prompt Optimization)

Uses Stanford's [DSPy MIPROv2](https://dspy-docs.vercel.app/) optimizer to automatically discover high-quality chain-of-thought prompts on the HotpotQA training split.

| Parameter | Value |
|-----------|-------|
| Optimizer | `MIPROv2` (`auto=medium`) |
| Max bootstrapped demos | 4 |
| Max labeled demos | 4 |
| Candidate prompts | 8 |
| Metric | LLM-as-a-Judge (binary) |

The optimized program is serialized to `optimized_programs/` and reused in evaluation.

### Stage 2 — ACE Offline (Single-Pass Evaluation)

Runs the optimized agent on the full test set in a single pass. Each prediction is scored for:

- **Accuracy** — Does the predicted answer match the gold answer? (LLM judge)
- **Faithfulness** — Is the answer grounded in the retrieved Wikipedia context?

### Stage 3 — ACE Online (Adaptive Self-Reflection)

Extends offline evaluation with an iterative reflection loop:

1. Run prediction → judge evaluates
2. If incorrect, `ACEReflector` generates a corrective "skill" based on the error
3. Skills accumulate in a **skillbook** appended to future prompts
4. Re-predict with augmented context (up to `max_reflections=3`)

---

## Results

Latest benchmark run on HotpotQA (single-hop, N=100 per stage):

| Stage | Accuracy | Description |
|-------|----------|-------------|
| **MIPROv2** | **86.0%** (86/100) | Optimized prompt, single-pass |
| **ACE Offline** | **78.0%** (78/100) | Faithfulness-constrained evaluation |
| **ACE Online** | **78.0%** (78/100) | With iterative self-reflection |

> Model: `claude-opus-4-6` · Metric: LLM-as-a-Judge (binary) · Dataset: HotpotQA single-hop (200 train / 100 test)

Full per-question results with predictions, expected answers, and verdicts are available in [`research_outputs/pipeline_results.txt`](research_outputs/pipeline_results.txt).

---

## Quickstart

### Prerequisites

- Python ≥ 3.10
- API keys for at least one LLM backend (see [Environment Variables](#environment-variables))

### Installation

```bash
# Clone the repository
git clone https://github.com/Dinesh0401/Wifipedia-agent.git
cd Wifipedia-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# Install in editable mode
pip install -e .
```

### Setup

```bash
# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Run

```bash
# Full pipeline (MIPROv2 + ACE offline + ACE online)
run-pipeline

# Or with Python directly
python -m scripts.run_pipeline
```

---

## Configuration

All settings are centralized in [`scripts/config.py`](scripts/config.py) and can be overridden via environment variables or CLI flags.

### CLI Options

```
usage: run_pipeline.py [-h] [--task {miprov2,ace,all,smoke}]
                       [--mode {offline,online,both}]
                       [--model {llmcontrols,claude,deepseek}]
                       [--train N] [--test N] [--no-mipro]

options:
  --task      Pipeline stage to run (default: all)
  --mode      ACE evaluation mode (default: both)
  --model     LLM backend to use (default: llmcontrols)
  --train N   Number of training samples (default: 200)
  --test N    Number of test samples (default: 100)
  --no-mipro  Skip MIPROv2 optimization, use existing program
```

### Task Configuration

Benchmark parameters are defined in [`benchmarks/tasks/hotpotqa.yaml`](benchmarks/tasks/hotpotqa.yaml):

```yaml
task: hotpotqa
data:
  filter:
    single_hop_only: true
    max_supporting_facts: 1
  seed: 42

retrieval:
  backend: wikipedia_api
  top_k: 3
  max_chars_per_doc: 2000

miprov2:
  auto: medium
  max_bootstrapped_demos: 4
  num_candidates: 8

ace:
  offline:
    enabled: true
  online:
    enabled: true
    max_reflections: 3
```

---

## Project Structure

```
Wifipedia-agent/
├── benchmarks/
│   └── tasks/
│       └── hotpotqa.yaml           # Benchmark task configuration
├── scripts/
│   ├── run_pipeline.py             # CLI entry point & orchestrator
│   ├── config.py                   # Centralized configuration (@dataclass)
│   ├── miprov2_pipeline.py         # DSPy MIPROv2 prompt optimization
│   ├── ace_pipeline.py             # ACE offline & online evaluation
│   ├── wiki_agent.py               # Wikipedia ReAct agent
│   ├── wiki_retriever.py           # Wikipedia document retrieval
│   ├── llm_client.py               # Unified async LLM client (multi-backend)
│   ├── llm_judge.py                # LLM-as-a-Judge binary evaluator
│   ├── llmcontrols_client.py       # LLMControls API client
│   ├── hotpotqa_loader.py          # HotpotQA dataset loader & splitter
│   ├── metrics.py                  # Accuracy + Wilson CI computation
│   ├── dspy_adapter.py             # DSPy integration helpers
│   └── dspy_llmcontrols.py         # DSPy ↔ LLMControls bridge
├── optimized_programs/             # Serialized MIPROv2 optimized programs
├── research_outputs/               # Predictions (JSONL), summaries, reports
├── results/                        # Final pipeline result snapshots
├── pyproject.toml                  # Project metadata & dependencies
└── README.md
```

---

## Usage

### Run the Full Pipeline

```bash
# All three stages with default settings
run-pipeline --task all --model claude
```

### Run Individual Stages

```bash
# MIPROv2 optimization only
run-pipeline --task miprov2 --train 200 --test 100

# ACE evaluation only (skip optimization, use existing program)
run-pipeline --task ace --mode both --no-mipro

# ACE offline only
run-pipeline --task ace --mode offline

# Quick smoke test (10 samples)
run-pipeline --task smoke
```

### Switch LLM Backends

```bash
# Use Anthropic Claude
run-pipeline --model claude

# Use DeepSeek via OpenRouter
run-pipeline --model deepseek

# Use LLMControls (default)
run-pipeline --model llmcontrols
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# ── LLM Backend (choose one or more) ────────────────────────
ACTIVE_MODEL=claude                  # llmcontrols | claude | deepseek

# LLMControls
LLMC_API_KEY=your-llmcontrols-key
LLMC_API_URL=https://api.llmcontrols.com/v1

# Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-...

# DeepSeek (via OpenRouter)
OPENROUTER_API_KEY=sk-or-...

# ── LLM Judge (separate endpoint to avoid self-eval bias) ───
JUDGE_API_KEY=your-judge-key
JUDGE_API_URL=https://judge.llmcontrols.com/v1

# ── Optional ─────────────────────────────────────────────────
MODEL_NAME=claude-opus-4-6            # Model name for LLMControls backend
OPENAI_API_KEY=sk-...                # May be needed by DSPy/LangChain internals
```

> **Important:** The judge endpoint (`JUDGE_API_KEY` / `JUDGE_API_URL`) should point to a **different** model or endpoint than the one generating answers to prevent self-evaluation bias.

---

## Key Components

### WikiAgent

LLM-powered agent that answers factual questions using retrieved Wikipedia passages. Enforces structured JSON output with analysis, answer, used sources, and confidence score. Answers are constrained to 1–5 words; returns `UNKNOWN` when context is insufficient.

### UnifiedLLMClient

Async LLM client that routes requests to the configured backend with:
- Automatic retry (3 attempts with exponential backoff)
- Backend-specific API formatting (LLMControls, Anthropic, OpenRouter)
- Runtime overrides for model, backend, API key, and URL

### LLMJudge

Binary evaluator using a separate LLM endpoint. Compares predicted answers against gold answers with semantic understanding (e.g., "Robert Erskine Childers" matches "Robert Erskine Childers DSC"). Always runs at `temperature=0.0` for deterministic verdicts.

### HotpotQALoader

Flexible dataset loader supporting:
- Local JSON files (`hotpot_train_v1.1.json`)
- HuggingFace datasets (`hotpot_qa`)
- Built-in demo fallback (10 samples)
- Filtering by question type (bridge/comparison) and difficulty

---

## Evaluation Methodology

### LLM-as-a-Judge

Rather than relying on classical metrics (exact match, F1), this pipeline uses an **LLM-as-a-Judge** approach:

1. A separate LLM endpoint receives the question, gold answer, and prediction
2. The judge determines if the prediction is semantically correct (binary: Yes/No)
3. Verdicts are aggregated into accuracy with Wilson score 95% confidence intervals

This approach handles paraphrasing, partial matches, and semantic equivalence that exact-match metrics miss.

### ACE Faithfulness

Each prediction is additionally evaluated for **faithfulness** — whether the answer is logically derivable from the retrieved Wikipedia context alone, without relying on the model's parametric knowledge.

### Output Format

Per-question results follow this structure:

```
Q001
Ques:         Were Scott Derrickson and Ed Wood of the same nationality?
Ans:          Yes
Expected Ans: yes
Correct:      YES
--------------------------------------------------------------------------------
```

Full JSONL prediction files with retrieval context, judge reasoning, and ACE faithfulness scores are saved to `research_outputs/`.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `dspy-ai` ≥ 2.5.0 | MIPROv2 prompt optimization |
| `langchain` ≥ 0.3.0 | Wikipedia retrieval chain |
| `langchain-community` ≥ 0.3.0 | Community integrations |
| `langchain-openai` ≥ 0.2.0 | OpenAI-compatible LLM wrapper |
| `datasets` ≥ 2.19.0 | HuggingFace dataset loading |
| `openai` ≥ 1.0.0 | API client for LLM backends |
| `aiohttp` ≥ 3.9.0 | Async HTTP for API calls |
| `numpy` ≥ 1.26.0 | Numerical computation |
| `pandas` ≥ 2.1.0 | Data manipulation |
| `scikit-learn` ≥ 1.4.0 | Metrics utilities |
| `python-dotenv` ≥ 1.0.0 | Environment variable loading |
| `pyyaml` ≥ 6.0.1 | YAML configuration parsing |
| `wikipedia` ≥ 1.4.0 | Wikipedia API access |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is for research purposes. See the repository for license details.

---

<p align="center">
  Built with <a href="https://dspy-docs.vercel.app/">DSPy</a> · <a href="https://python.langchain.com/">LangChain</a> · <a href="https://hotpotqa.github.io/">HotpotQA</a>
</p>
