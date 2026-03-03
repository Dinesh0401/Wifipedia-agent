# =============================================================================
# wiki_agent.py  -- Wikipedia ReAct Agent (LangChain + LLMControls)
# Real retrieval | Real LLM | Strict JSON output | ACE-ready
# =============================================================================

import re
import json
import asyncio
import logging
from collections import Counter
from datetime import datetime
from typing import List, Dict, Any, Optional

from scripts.wiki_retriever import WikipediaRetriever
from scripts.llm_client import UnifiedLLMClient
from scripts.llm_judge import LLMJudge
from scripts.config import cfg

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert multi-hop question answering system.

TASK: Answer using ONLY the retrieved Wikipedia passages. Follow the reasoning chain carefully.

STEP-BY-STEP REASONING PROCESS:
1. Identify the question type:
   - BRIDGE: requires finding an intermediate entity first, then answering about it
     Example: "What city is the HQ of the hotel chain owned by [person/family]?"
     Chain: Find which hotel chain → Find that chain's HQ city
   - COMPARISON: requires extracting two values and comparing them
     Example: "Which was founded earlier, X or Y?"
     Chain: Find founding year of X → Find founding year of Y → Compare
   - DIRECT: answer is stated directly in a passage

2. For BRIDGE questions:
   - Step A: Identify the intermediate entity from the passages (e.g. the name of the company, film, person)
   - Step B: Find the final answer about that intermediate entity in the passages
   - Do NOT guess; the answer MUST be in the retrieved texts

3. For COMPARISON questions:
   - Extract the exact dates/numbers for BOTH items from the passages
   - Compare them explicitly before writing the answer
   - State which is larger/earlier/older based on the numbers

4. For YES/NO questions:
   - Find the relevant fact in the passages
   - Answer ONLY "yes" or "no"

STRICT RULES:
- Answer must be 1 to 5 words maximum.
- Extract answer exactly as it appears in the text.
- NEVER answer UNKNOWN. Give your best answer from the passages.
- Answer field contains ONLY the answer. No explanation.

Return ONLY valid JSON:
{
    "analysis": "step-by-step chain showing how you found the answer from the passages",
    "answer": "short exact answer",
    "used_titles": ["titles of passages used"],
    "confidence": 0.0-1.0
}"""


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
      2. reason()   -> call LLM API with strict JSON prompt
      3. evaluate() -> LLM-as-a-Judge semantic evaluation
      4. record()   -> immutable JSONL record ready for ACE / MIPROv2
    """

    def __init__(self, skill_prefix: str = "", optimized_prompt: str = ""):
        self.retriever = WikipediaRetriever()
        self.client = UnifiedLLMClient()
        self.judge = LLMJudge()
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

        docs = await self.retriever.retrieve_async(question)
        if not docs:
            logger.warning(f"[{sample['_id']}] Zero retrieval results")
            docs = []

        # --- Retrieval recall tracking ---
        gold_titles = [sf[0] for sf in sample.get("supporting_facts", [])]
        retrieved_titles = [d["title"] for d in docs]
        recall_hit = any(gt.lower() in rt.lower()
                         for gt in gold_titles
                         for rt in retrieved_titles)

        # --- LLM reranking ---
        if len(docs) > 5:
            candidate_docs = docs[:10]
            rerank_prompt = f"""Question: {question}

Documents:
{chr(10).join(f"[{i+1}] {d['title']}: {d['content'][:300]}" for i, d in enumerate(candidate_docs))}

Which 5 document numbers best answer this question?
Return ONLY JSON: {{"top": [1, 2, 3, 4, 5]}}"""
            try:
                raw = await self.client.generate(rerank_prompt)
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                indices = json.loads(match.group())["top"]
                docs = [candidate_docs[i-1] for i in indices if 0 < i <= len(candidate_docs)]
            except Exception:
                docs = candidate_docs[:5]
        docs = docs[:5]

        # --- Two-pass bridge retrieval ---
        # Ask the LLM which intermediate entity (bridge) still needs a Wikipedia lookup.
        # This is the #1 fix for HotpotQA bridge questions.
        bridge_prompt = f"""Question: {question}

Retrieved passage titles so far: {[d['title'] for d in docs]}
Passage snippets:
{chr(10).join(f"[{i+1}] {d['title']}: {d['content'][:250]}" for i, d in enumerate(docs))}

If this question requires finding an intermediate entity FIRST (a bridge hop), name that entity.
If all needed facts are already present, respond with "none".
Return ONLY JSON: {{"entity": "entity name or none"}}"""
        try:
            raw_bridge = await self.client.generate(bridge_prompt)
            match_bridge = re.search(r"\{.*\}", raw_bridge, re.DOTALL)
            bridge_entity = json.loads(match_bridge.group()).get("entity", "none").strip()
            if bridge_entity.lower() not in ("none", "", "n/a") and len(bridge_entity) > 2:
                logger.info(f"[Bridge] Second-hop lookup: '{bridge_entity}'")
                second_hop_docs = await self.retriever.retrieve_async(bridge_entity)
                existing_titles = {d["title"].lower() for d in docs}
                added = 0
                for d in second_hop_docs:
                    if d["title"].lower() not in existing_titles and added < 3:
                        docs.append(d)
                        existing_titles.add(d["title"].lower())
                        added += 1
        except Exception as e:
            logger.debug(f"Bridge retrieval skipped: {e}")

        context = self.retriever.build_context(docs)
        prediction = await self._reason_with_voting(question, context)

        judgment = await self.judge.judge(
            question=question,
            gold_answer=gold_ans,
            prediction=prediction["answer"],
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
                "recall_hit": recall_hit,
            },
            "prediction": prediction,
            "metrics": {
                "judge_verdict": judgment["verdict"],
                "judge_correct": judgment["correct"],
                "judge_reasoning": judgment["reasoning"],
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
                    verdict = results[idx]["metrics"]["judge_verdict"]
                    correct = results[idx]["metrics"]["judge_correct"]
                    logger.info(
                        f"[{idx+1}/{len(samples)}] {sample['_id']}"
                        f" verdict={verdict} correct={correct}"
                    )
                except Exception as e:
                    logger.error(f"[{idx+1}] FAILED {sample.get('_id', '?')}: {e}")
                    results[idx] = self._error_record(sample, str(e))

        await asyncio.gather(*[_process(i, s) for i, s in enumerate(samples)])
        return [r for r in results if r is not None]

    async def _reason_with_voting(self, question: str, context: str) -> dict:
        from collections import Counter

        # Round 1: 7 independent calls for stronger majority signal
        tasks = [self._reason(question, context) for _ in range(7)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid = [r for r in results if isinstance(r, dict)]

        if not valid:
            return {
                "answer": self._best_effort_answer(question, context),
                "analysis": "fallback",
                "used_titles": [],
                "confidence": 0.0,
            }

        # Normalize before counting (yes/YES/Yes all count the same)
        def _norm(s: str) -> str:
            return s.lower().strip().rstrip(".")

        counts = Counter(
            _norm(r["answer"])
            for r in valid
            if _norm(r.get("answer", "")) not in ("unknown", "")
        )
        if not counts:
            return {
                "answer": self._best_effort_answer(question, context),
                "analysis": "fallback",
                "used_titles": [],
                "confidence": 0.0,
            }
        best, best_count = counts.most_common(1)[0]

        # Strong majority: 4+ out of 7 → return it directly
        if best_count >= 4:
            for r in valid:
                if _norm(r["answer"]) == best:
                    return r

        # Round 2: verifier with full context (not truncated to 2000)
        verify_prompt = f"""Question: {question}

Wikipedia Context:
{context[:8000]}

Multiple reasoning attempts returned these candidate answers (with vote counts):
{[f"{ans} ({cnt} votes)" for ans, cnt in counts.most_common()]}

Carefully re-read the context and determine the single correct answer.
For a bridge question: identify the intermediate entity first, then find the final answer.
For a comparison question: extract both values then compare.
Return ONLY JSON: {{"answer": "correct answer", "reasoning": "step-by-step chain"}}"""

        try:
            raw = await self.client.generate(verify_prompt)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            verified = json.loads(match.group())
            answer = str(verified.get("answer", "")).strip()
            if answer:
                return {
                    "answer": answer,
                    "analysis": verified.get("reasoning", ""),
                    "used_titles": valid[0].get("used_titles", []),
                    "confidence": 0.95,
                }
        except Exception:
            pass

        for r in valid:
            if _norm(r["answer"]) == best:
                return r
        return valid[0]

    async def _reason(self, question: str, context: str) -> Dict[str, Any]:
        prompt = f"{self.optimized_prompt}{self.skill_prefix}{SYSTEM_PROMPT}\n\n{build_user_prompt(question, context)}"
        raw_output = await self.client.generate(prompt)
        try:
            return self._parse_json(raw_output)
        except ValueError as e:
            logger.warning(f"JSON parse failed: {e}. Returning degraded prediction.")
            return {
                "analysis": "parse_error",
                "answer": self._best_effort_answer(question, context),
                "used_titles": [],
                "confidence": 0.0,
                "raw_output": raw_output[:300],
            }

    @staticmethod
    def _best_effort_answer(question: str, context: str) -> str:
        q_lower = question.lower().strip()
        if q_lower.startswith(
            ("is ", "are ", "was ", "were ", "do ", "does ", "did ",
             "can ", "could ", "should ", "has ", "have ", "had ")
        ):
            return "no"
        for line in context.splitlines():
            line = line.strip()
            if not line or line.startswith("TITLE:") or line.startswith("---"):
                continue
            words = re.findall(r"[A-Za-z0-9'\-]+", line)
            if words:
                return " ".join(words[:5])
        return "no"

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
            "metrics": {"judge_verdict": "No", "judge_correct": False, "judge_reasoning": "error"},
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
