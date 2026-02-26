# =============================================================================
# wiki_agent.py  -- Wikipedia ReAct Agent (LangChain + LLMControls)
# Real retrieval | Real LLM | Strict JSON output | ACE-ready
# =============================================================================

import re
import json
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from scripts.wiki_retriever import WikipediaRetriever
from scripts.llm_client import UnifiedLLMClient
from scripts.metrics import token_precision_recall_f1, supporting_fact_f1
from scripts.config import cfg

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a research assistant solving multi-hop Wikipedia questions.

TASK: Given a question and retrieved Wikipedia passages, provide a precise answer.

RULES:
1. Answer ONLY from the retrieved passages. Do NOT use prior knowledge.
2. Bridge reasoning: identify Entity1 -> bridge entity -> Entity2.
3. Keep the answer SHORT (1-5 words). Match expected HotpotQA answer format.
4. If the passages do not contain the answer, set answer to "UNKNOWN".
5. Return ONLY valid JSON. No markdown, no explanation outside JSON.

OUTPUT FORMAT:
{
  "analysis": "brief chain-of-thought (max 2 sentences)",
  "answer": "short final answer",
  "used_titles": ["list of Wikipedia titles you relied on"],
  "confidence": 0.0-1.0
}
"""


def build_user_prompt(question: str, context: str) -> str:
    return f"""QUESTION:
{question}

RETRIEVED WIKIPEDIA PASSAGES:
{context}

Return ONLY the JSON object described in the instructions."""


class WikiAgent:
    """
    Single-sample Wikipedia QA agent.

    Pipeline per sample:
      1. retrieve() -> real Wikipedia passages via LangChain
      2. reason()   -> call LLMControls API with strict JSON prompt
      3. evaluate() -> EM + F1 + supporting-fact F1
      4. record()   -> immutable JSONL record ready for ACE / MIPROv2
    """

    def __init__(self, skill_prefix: str = "", optimized_prompt: str = ""):
        self.retriever = WikipediaRetriever()
        self.client = UnifiedLLMClient()
        self.skill_prefix = skill_prefix
        self.optimized_prompt = optimized_prompt
        self._run_metadata = {
            "model": cfg.get_display_model_name(),
            "temperature": cfg.temperature,
            "wiki_top_k": cfg.wiki_top_k,
            "timestamp": datetime.now().isoformat(),
        }

    async def run_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        question = sample["question"]
        gold_ans = sample["answer"]
        gold_facts = sample.get("supporting_facts", [])

        docs = await self.retriever.retrieve_async(question)
        if not docs:
            logger.warning(f"[{sample['_id']}] Zero retrieval results")
            docs = []

        context = self.retriever.build_context(docs)
        prediction = await self._reason(question, context)

        prec, rec, f1 = token_precision_recall_f1(prediction["answer"], gold_ans)
        sf_f1 = supporting_fact_f1(
            prediction.get("used_titles", []), gold_facts
        )

        return {
            "metadata": self._run_metadata,
            "task": {
                "id": sample["_id"],
                "question": question,
                "gold_answer": gold_ans,
                "question_type": sample.get("type", "bridge"),
                "question_level": sample.get("level", "medium"),
            },
            "retrieval": {
                "docs": docs,
                "raw_context": context,
                "n_docs": len(docs),
            },
            "prediction": prediction,
            "metrics": {
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "supporting_fact_f1": sf_f1,
            },
        }

    async def run_batch(
        self,
        samples: List[Dict[str, Any]],
        concurrency: int = 3,
    ) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(concurrency)
        results: List[Optional[Dict[str, Any]]] = [None] * len(samples)

        async def _process(idx: int, sample: Dict[str, Any]) -> None:
            async with semaphore:
                try:
                    results[idx] = await self.run_sample(sample)
                    prec = results[idx]["metrics"]["precision"]
                    rec = results[idx]["metrics"]["recall"]
                    f1 = results[idx]["metrics"]["f1"]
                    logger.info(
                        f"[{idx+1}/{len(samples)}] {sample['_id']}"
                        f" P={prec:.2f} R={rec:.2f} F1={f1:.2f}"
                    )
                except Exception as e:
                    logger.error(f"[{idx+1}] FAILED {sample.get('_id', '?')}: {e}")
                    results[idx] = self._error_record(sample, str(e))

        await asyncio.gather(*[_process(i, s) for i, s in enumerate(samples)])
        return [r for r in results if r is not None]

    async def _reason(self, question: str, context: str) -> Dict[str, Any]:
        prompt = f"{self.optimized_prompt}{self.skill_prefix}{SYSTEM_PROMPT}\n\n{build_user_prompt(question, context)}"
        raw_output = await self.client.generate(prompt)
        try:
            return self._parse_json(raw_output)
        except ValueError as e:
            logger.warning(f"JSON parse failed: {e}. Returning degraded prediction.")
            return {
                "analysis": "parse_error",
                "answer": "UNKNOWN",
                "used_titles": [],
                "confidence": 0.0,
                "raw_output": raw_output[:300],
            }

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        text = re.sub(r"```(?:json)?\s*", "", text).strip()
        text = re.sub(r"```\s*$", "", text).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in output: {text[:200]!r}")
        parsed = json.loads(match.group())
        for field in ("analysis", "answer", "used_titles"):
            if field not in parsed:
                raise ValueError(f"Missing required field: {field!r}")
        if not isinstance(parsed["answer"], str) or not parsed["answer"].strip():
            raise ValueError("answer must be a non-empty string")
        return parsed

    @staticmethod
    def _error_record(sample: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        return {
            "metadata": {"error": error_msg},
            "task": {
                "id": sample.get("_id", "?"),
                "question": sample.get("question", ""),
                "gold_answer": sample.get("answer", ""),
            },
            "retrieval": {"docs": [], "raw_context": "", "n_docs": 0},
            "prediction": {
                "analysis": "error",
                "answer": "UNKNOWN",
                "used_titles": [],
                "confidence": 0.0,
            },
            "metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "supporting_fact_f1": 0.0},
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    demo = {
        "_id": "test001",
        "question": "What city is the headquarters of the hotel chain owned by the Oberoi family?",
        "answer": "Delhi",
        "supporting_facts": [["EIH Limited", 0]],
        "type": "bridge",
        "level": "easy",
    }

    async def main():
        agent = WikiAgent()
        record = await agent.run_sample(demo)
        print(json.dumps(record, indent=2, default=str))

    asyncio.run(main())
