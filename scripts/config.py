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
load_dotenv(_project_root / ".env")


@dataclass
class PipelineConfig:
    # -- API -----------------------------------------------------------------
    llmc_api_key: str     = field(default_factory=lambda: os.getenv("LLMC_API_KEY", ""))
    llmc_api_url: str     = field(default_factory=lambda: os.getenv("LLMC_API_URL", ""))
    openai_api_key: str   = field(default_factory=lambda: os.getenv("LLMC_API_KEY", ""))
    openai_base_url: str  = field(default_factory=lambda: os.getenv(
        "OPENAI_BASE_URL", "https://api.llmcontrols.ai/api/v1"
    ))
    model_name: str       = field(default_factory=lambda: os.getenv("MODEL_NAME", "gpt-4o-mini"))

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
        if not self.llmc_api_key:
            raise ValueError("LLMC_API_KEY not set. Add it to .env")
        if not self.llmc_api_url:
            raise ValueError("LLMC_API_URL not set. Add it to .env")


# -- Singleton ---------------------------------------------------------------
cfg = PipelineConfig()
