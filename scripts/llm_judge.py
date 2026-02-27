# =============================================================================
# llm_judge.py  -- LLM-as-a-Judge Evaluation
# Replaces all classical metrics (F1, precision, recall, exact_match)
# Deterministic (temperature=0), uses a different model than the task model
# =============================================================================

import re
import json
import logging
from typing import Dict

from scripts.llm_client import UnifiedLLMClient
from scripts.config import cfg

logger = logging.getLogger(__name__)


class LLMJudge:

    PROMPT = """You are evaluating a question-answering system.

Question:
{question}

Gold Answer:
{gold_answer}

Predicted Answer:
{prediction}

Is the predicted answer semantically correct?

Rules:
- Paraphrases count as correct.
- Minor formatting differences count as correct.
- If UNKNOWN or clearly wrong, answer No.

Return ONLY valid JSON:

{{
  "verdict": "Yes" or "No",
  "reasoning": "one short sentence"
}}
"""

    def __init__(self):
        # Judge uses a separate LLMControls endpoint (different from task model)
        self.client = UnifiedLLMClient(
            backend_override="llmcontrols",
            api_key_override=cfg.judge_api_key,
            api_url_override=cfg.judge_api_url,
        )

    async def judge(self, question: str, gold_answer: str, prediction: str) -> dict:
        prompt = self.PROMPT.format(
            question=question,
            gold_answer=gold_answer,
            prediction=prediction,
        )

        raw = await self.client.generate(prompt, temperature=0.0)

        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            result = json.loads(match.group())
            verdict = str(result.get("verdict", "No")).strip()
            correct = verdict.lower() == "yes"
            return {
                "verdict": "Yes" if correct else "No",
                "correct": correct,
                "reasoning": result.get("reasoning", ""),
            }

        except Exception:
            logger.warning(f"LLM judge parse error. Raw: {raw[:200]}")
            return {
                "verdict": "No",
                "correct": False,
                "reasoning": "parse_error",
            }

    def judge_sync(self, question: str, gold_answer: str, prediction: str) -> dict:
        """Blocking wrapper for use in DSPy metrics and other sync contexts."""
        prompt = self.PROMPT.format(
            question=question,
            gold_answer=gold_answer,
            prediction=prediction,
        )

        raw = self.client.generate_sync(prompt, temperature=0.0)

        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            result = json.loads(match.group())
            verdict = str(result.get("verdict", "No")).strip()
            correct = verdict.lower() == "yes"
            return {
                "verdict": "Yes" if correct else "No",
                "correct": correct,
                "reasoning": result.get("reasoning", ""),
            }

        except Exception:
            logger.warning(f"LLM judge parse error. Raw: {raw[:200]}")
            return {
                "verdict": "No",
                "correct": False,
                "reasoning": "parse_error",
            }
