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

    PROMPT = """You are a strict but fair answer evaluator.

Question: {question}
Gold Answer: {gold_answer}
Predicted Answer: {prediction}

Evaluation rules:
- Exact match -> correct = true, score = 1.0
- Valid paraphrase or abbreviation -> correct = true, score = 0.9
- Same entity different format (e.g. "New York" vs "NYC") -> correct = true, score = 0.95
- Partially correct (right topic, wrong detail) -> correct = false, score = 0.4
- Wrong or unrelated -> correct = false, score = 0.0

Return ONLY valid JSON:
{
    "correct": true or false,
    "score": 0.0 to 1.0,
    "reasoning": "one sentence"
}"""

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
            correct = bool(result.get("correct", False))
            score = float(result.get("score", 0.0))
            return {
                "verdict": "Yes" if correct else "No",
                "correct": correct,
                "score": score,
                "reasoning": result.get("reasoning", ""),
            }
        except Exception:
            logger.warning(f"LLM judge parse error. Raw: {raw[:200]}")
            return {
                "verdict": "No",
                "correct": False,
                "score": 0.0,
                "reasoning": "parse_error",
            }

    def judge_sync(self, question: str, gold_answer: str, prediction: str) -> dict:
        """Blocking wrapper for use in DSPy metrics and other sync contexts."""
        import asyncio
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except Exception:
            pass
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.judge(question, gold_answer, prediction)
        )
