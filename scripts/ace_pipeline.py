# =============================================================================
# ace_pipeline.py  -- ACE Benchmark Evaluation (Offline + Online)
# ACE = Agentic Context Engine -- measures faithfulness + accuracy
# Offline: run once over test set | Online: iterative adaptation loop
# =============================================================================

import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import dspy

from scripts.wiki_agent import WikiAgent
from scripts.wiki_retriever import WikipediaRetriever
from scripts.llm_client import UnifiedLLMClient
from scripts.metrics import (
    compute_metrics, print_metrics, wilson_ci,
)
from scripts.hotpotqa_loader import HotpotQALoader
from scripts.dspy_adapter import configure_dspy
from scripts.config import cfg

logger = logging.getLogger(__name__)


# -- ACE Faithfulness Signature (DSPy) --------------------------------------

class ACEFaithfulness(dspy.Signature):
    """Evaluate whether a prediction is grounded in the provided context."""
    question = dspy.InputField(desc="The original question")
    context = dspy.InputField(desc="Retrieved Wikipedia passages")
    prediction = dspy.InputField(desc="The model's predicted answer")
    faithful = dspy.OutputField(desc="true or false")
    reasoning = dspy.OutputField(desc="Brief explanation of the faithfulness judgement")


# -- ACE Skill Reflector (Online mode) --------------------------------------

class ACEReflector:
    """After each wrong prediction, generate a reflective skill for the skillbook."""

    def __init__(self):
        self.skillbook: List[str] = []
        self.client = UnifiedLLMClient()

    async def reflect(
        self,
        question: str,
        gold_answer: str,
        wrong_prediction: str,
        context: str,
    ) -> str:
        prompt = (
            f"A model answered '{wrong_prediction}' but the correct answer was '{gold_answer}'.\n"
            f"Question: {question}\n"
            f"Context: {context[:500]}\n\n"
            "In ONE sentence, state what retrieval or reasoning heuristic would have "
            "prevented this mistake. Start with 'When ...'"
        )
        try:
            skill = await self.client.generate(prompt)
            skill = skill.strip().split("\n")[0][:200]
            self.skillbook.append(skill)
            logger.info(f"[ACE Reflect] New skill: {skill}")
            return skill
        except Exception as e:
            logger.warning(f"Reflection failed: {e}")
            return ""

    def get_skill_prefix(self) -> str:
        if not self.skillbook:
            return ""
        skills = "\n".join(f"  - {s}" for s in self.skillbook[-5:])
        return f"LEARNED SKILLS:\n{skills}\n\n"


# -- ACE Faithfulness Evaluator ---------------------------------------------

class ACEFaithfulnessEvaluator:
    """Uses DSPy ChainOfThought to judge faithfulness of predictions."""

    def __init__(self):
        configure_dspy(temperature=0.0, max_tokens=150)
        self.program = dspy.ChainOfThought(ACEFaithfulness)

    def evaluate(
        self, question: str, context: str, prediction: str
    ) -> Dict[str, Any]:
        try:
            result = self.program(
                question=question,
                context=context[:1500],
                prediction=prediction,
            )
            faithful_bool = "true" in result.faithful.lower()
            return {
                "faithful": faithful_bool,
                "reasoning": result.reasoning,
            }
        except Exception as e:
            logger.warning(f"ACE faithfulness eval failed: {e}")
            return {"faithful": False, "reasoning": f"eval_error: {e}"}


# -- Main ACE Runner --------------------------------------------------------

class ACERunner:
    """
    Runs the full ACE benchmark in offline or online mode.
    Offline: agent.run_batch(test_samples) -> metrics -> save
    Online:  iterative loop with reflection after wrong predictions
    """

    def __init__(self, online_mode: bool = False, optimized_program_path: str = None):
        self.online_mode = online_mode
        self.optimized_prompt = self._load_optimized_prompt(optimized_program_path)
        self.agent = WikiAgent(optimized_prompt=self.optimized_prompt)
        self.reflector = ACEReflector() if online_mode else None
        self.faithfulness = ACEFaithfulnessEvaluator()

    @staticmethod
    def _load_optimized_prompt(path: str) -> str:
        """Load the optimized prompt/demos from MIPROv2 JSON output."""
        if not path:
            return ""
        try:
            import json
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Extract any instruction or demo text from the saved program
            parts = []
            # DSPy saves programs with various structures; extract what we can
            if isinstance(data, dict):
                for key, val in data.items():
                    if isinstance(val, str) and len(val) > 10:
                        parts.append(val)
                    elif isinstance(val, dict):
                        for subkey, subval in val.items():
                            if isinstance(subval, str) and len(subval) > 10:
                                parts.append(f"{subkey}: {subval}")
            if parts:
                prompt = "OPTIMIZED INSTRUCTIONS:\n" + "\n".join(parts[:5]) + "\n\n"
                logger.info(f"Loaded optimized prompt from {path} ({len(prompt)} chars)")
                return prompt
        except Exception as e:
            logger.warning(f"Could not load optimized program from {path}: {e}")
        return ""

    # -- Offline evaluation --------------------------------------------------
    async def run_offline(
        self, test_samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        logger.info(f"[ACE Offline] Running {len(test_samples)} samples ...")
        predictions = await self.agent.run_batch(test_samples, concurrency=3)
        predictions = self._attach_faithfulness(predictions)
        return self._save_and_summarise(predictions, mode="offline")

    # -- Online evaluation ---------------------------------------------------
    async def run_online(
        self,
        test_samples: List[Dict[str, Any]],
        max_reflections: int = None,
    ) -> Dict[str, Any]:
        max_ref = max_reflections or cfg.ace_max_reflections
        logger.info(f"[ACE Online] {len(test_samples)} samples | max_reflections={max_ref}")

        predictions = []
        reflections = 0

        for i, sample in enumerate(test_samples):
            # Create agent with current skills AND optimized prompt injected
            skill_prefix = self.reflector.get_skill_prefix() if self.reflector else ""
            agent = WikiAgent(
                skill_prefix=skill_prefix,
                optimized_prompt=self.optimized_prompt,
            )
            record = await agent.run_sample(sample)

            if not record["metrics"]["judge_correct"] and reflections < max_ref:
                skill = await self.reflector.reflect(
                    question=sample["question"],
                    gold_answer=sample["answer"],
                    wrong_prediction=record["prediction"].get("answer", "UNKNOWN"),
                    context=record["retrieval"]["raw_context"],
                )
                reflections += 1
                record["online_skill"] = skill

            record = self._attach_faithfulness_single(record)
            predictions.append(record)

            score = record["metrics"]["judge_verdict"]
            correct = record["metrics"]["judge_correct"]
            logger.info(
                f"[{i+1}/{len(test_samples)}] verdict={score} correct={correct} "
                f"faithful={record['ace']['faithful']}"
            )

        result = self._save_and_summarise(predictions, mode="online")
        result["online_stats"] = {
            "reflections_used": reflections,
            "skillbook_size": len(self.reflector.skillbook),
            "skillbook": self.reflector.skillbook,
        }
        return result

    # -- Faithfulness attachment ---------------------------------------------
    def _attach_faithfulness(
        self, records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        for r in records:
            self._attach_faithfulness_single(r)
        return records

    def _attach_faithfulness_single(
        self, record: Dict[str, Any]
    ) -> Dict[str, Any]:
        faith = self.faithfulness.evaluate(
            question=record["task"]["question"],
            context=record["retrieval"].get("raw_context", ""),
            prediction=record["prediction"].get("answer", ""),
        )
        record["ace"] = {
            "faithful": faith["faithful"],
            "ace_reasoning": faith["reasoning"],
        }
        return record

    # -- Save + summarise (LLM-as-a-Judge JSON schema) ---------------------------
    def _save_and_summarise(
        self, predictions: List[Dict[str, Any]], mode: str
    ) -> Dict[str, Any]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"hotpotqa_ace_{mode}_{timestamp}"

        jsonl_path = cfg.output_dir / f"{base}_predictions.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in predictions:
                f.write(json.dumps(r, default=str) + "\n")

        metric_records = [{
            "judge_correct": p["metrics"]["judge_correct"],
        } for p in predictions]
        summary = compute_metrics(metric_records, split=f"test/{mode}")
        print_metrics(summary)

        faithful_vals = [p.get("ace", {}).get("faithful", False) for p in predictions]
        ace_rate = float(np.mean(faithful_vals)) if faithful_vals else 0.0
        ace_lo, ace_hi = wilson_ci(ace_rate, len(predictions))

        # LLM-as-a-Judge output schema
        full_summary = {
            "benchmark": "hotpotqa",
            "model": cfg.get_display_model_name(),
            "timestamp": timestamp,
            "samples_evaluated": len(predictions),
            "summary_metrics": {
                "yes_count": summary["yes_count"],
                "no_count": summary["no_count"],
                "accuracy": summary["accuracy"],
                "accuracy_pct": summary["accuracy_pct"],
                "wilson_ci95": summary["wilson_ci95"],
                "ace_faithfulness": round(ace_rate, 4),
                "ace_faithfulness_pct": f"{ace_rate * 100:.1f}%",
            },
            "configuration": {
                "split": "test",
                "epochs": 1,
                "temperature": cfg.temperature,
                "max_tokens": 2048,
                "skip_adaptation": False,
                "split_ratio": 0.8,
                "online_mode": mode == "online",
                "prompt_version": "v1",
                "evaluation_mode": "online" if mode == "online" else "offline_train_test_split",
            },
        }
        summary_path = cfg.output_dir / f"{base}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(full_summary, f, indent=2, default=str)

        print(f"\nOutputs:")
        print(f"  JSONL   -> {jsonl_path}")
        print(f"  Summary -> {summary_path}")

        # Attach per-sample verdicts for pipeline output formatting
        full_summary["per_sample_verdicts"] = [
            p["metrics"]["judge_correct"] for p in predictions
        ]
        full_summary["per_sample_faithful"] = [
            p.get("ace", {}).get("faithful", False) for p in predictions
        ]
        # Attach per-sample records with question/prediction/gold_answer/correct
        full_summary["per_sample_records"] = [
            {
                "question": p["task"]["question"],
                "prediction": p["prediction"].get("answer", "UNKNOWN"),
                "gold_answer": p["task"]["gold_answer"],
                "correct": p["metrics"]["judge_correct"],
            }
            for p in predictions
        ]

        return full_summary


# -- Standalone runner -------------------------------------------------------

async def main(online: bool = False):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    loader = HotpotQALoader()
    _, test = loader.load_train_test(max_train=1, max_test=cfg.max_test_samples)
    runner = ACERunner(online_mode=online)
    if online:
        result = await runner.run_online(test)
    else:
        result = await runner.run_offline(test)
    m = result["summary_metrics"]
    print(f"\nACE benchmark complete.")
    print(f"  Accuracy : {m['accuracy_pct']}  CI95 {m['wilson_ci95']}")
    print(f"  Yes/No   : {m['yes_count']}/{m['no_count']}")
    print(f"  Faithful : {m['ace_faithfulness_pct']}")


if __name__ == "__main__":
    import sys
    online_flag = "--online" in sys.argv
    asyncio.run(main(online=online_flag))
