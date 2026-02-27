# =============================================================================
# config.py  -- Central configuration for Wiki Agent Research Pipeline
# =============================================================================

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env", override=True)

# Valid model backend names
VALID_MODELS = ("llmcontrols", "claude", "deepseek")


@dataclass
class PipelineConfig:
    # -- API keys ------------------------------------------------------------
    llmc_api_key: str        = field(default_factory=lambda: os.getenv("LLMC_API_KEY", ""))
    llmc_api_url: str        = field(default_factory=lambda: os.getenv("LLMC_API_URL", ""))
    anthropic_api_key: str   = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    openrouter_api_key: str  = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))

    # -- Judge API (separate LLMControls endpoint for LLM-as-a-Judge) ---------
    judge_api_key: str       = field(default_factory=lambda: os.getenv("JUDGE_API_KEY", ""))
    judge_api_url: str       = field(default_factory=lambda: os.getenv("JUDGE_API_URL", ""))

    # -- Active model (set via --model CLI flag or env) ----------------------
    active_model: str = field(default_factory=lambda: os.getenv("ACTIVE_MODEL", "llmcontrols"))

    # -- Model identifiers ---------------------------------------------------
    claude_model: str       = "claude-opus-4-6"
    deepseek_model: str     = "deepseek/deepseek-chat-v3-0324"
    model_name: str         = field(default_factory=lambda: os.getenv("MODEL_NAME", "gpt-4o-mini"))

    # -- Retrieval -----------------------------------------------------------
    wiki_top_k: int       = 3
    wiki_chars_max: int   = 2000
    temperature: float    = 0.0

    # -- Dataset -------------------------------------------------------------
    hotpot_train_file: str  = "hotpot_train_v1.1.json"
    hotpot_dev_file: str    = "hotpot_dev_distractor_v1.json"
    max_train_samples: int  = 200
    max_test_samples: int   = 100
    random_seed: int        = 42

    # -- MIPROv2 -------------------------------------------------------------
    miprov2_max_bootstrapped: int  = 4
    miprov2_max_labeled: int       = 4
    miprov2_num_candidates: int    = 8
    miprov2_auto: str              = "medium"

    # -- ACE -----------------------------------------------------------------
    ace_max_reflections: int   = 3
    ace_online_mode: bool      = False

    # -- Paths ---------------------------------------------------------------
    output_dir: Path     = field(default_factory=lambda: _project_root / "research_outputs")
    opt_dir: Path        = field(default_factory=lambda: _project_root / "optimized_programs")

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True)
        self.opt_dir.mkdir(exist_ok=True)
        if self.active_model not in VALID_MODELS:
            raise ValueError(
                f"ACTIVE_MODEL must be one of {VALID_MODELS}, got '{self.active_model}'"
            )
        # Validate that the chosen model has its API key
        if self.active_model == "llmcontrols":
            if not self.llmc_api_key or not self.llmc_api_url:
                raise ValueError("LLMC_API_KEY and LLMC_API_URL must be set for llmcontrols model")
        elif self.active_model == "claude":
            if not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY must be set for claude model")
        elif self.active_model == "deepseek":
            if not self.openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY must be set for deepseek model")

    def get_display_model_name(self) -> str:
        """Return a human-readable model name for the active backend."""
        if self.active_model == "llmcontrols":
            return f"llmcontrols/{self.model_name}"
        elif self.active_model == "claude":
            return self.claude_model
        elif self.active_model == "deepseek":
            return self.deepseek_model
        return self.active_model


# -- Singleton ---------------------------------------------------------------
cfg = PipelineConfig()
