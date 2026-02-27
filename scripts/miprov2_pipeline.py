# =============================================================================
# miprov2_pipeline.py  -- DSPy MIPROv2 Prompt Optimization
# Optimizes the reasoning prompt on the train split.
# Saves optimized program JSON for use in ACE evaluation.
# =============================================================================

import os
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import dspy

from scripts.llm_judge import LLMJudge
from scripts.metrics import compute_metrics, print_metrics
from scripts.hotpotqa_loader import HotpotQALoader
from scripts.wiki_retriever import WikipediaRetriever
from scripts.dspy_adapter import configure_dspy
from scripts.config import cfg

logger = logging.getLogger(__name__)


# -- DSPy signatures --------------------------------------------------------

class BridgeQA(dspy.Signature):
    """Answer a Wikipedia factual question given retrieved context."""
    question = dspy.InputField(desc="A factual question")
    context = dspy.InputField(desc="Retrieved Wikipedia passages (may contain distractors)")
    answer = dspy.OutputField(desc="Short factual answer (1-5 words)")


class BridgeQAWithReasoning(dspy.Signature):
    """Solve a Wikipedia factual question with chain-of-thought."""
    question = dspy.InputField(desc="Factual question")
    context = dspy.InputField(desc="Retrieved Wikipedia passages")
    reasoning = dspy.OutputField(desc="Chain-of-thought reasoning to the answer")
    answer = dspy.OutputField(desc="Concise final answer (1-5 words)")


# -- HotpotQA metric for DSPy (LLM-as-a-Judge) --------------------------------

_judge = LLMJudge()

def hotpotqa_metric(example: dspy.Example, pred, trace=None) -> float:
    gold = example.answer
    pred_ans = getattr(pred, "answer", "") or ""
    result = _judge.judge_sync(
        question=example.question,
        gold_answer=gold,
        prediction=pred_ans,
    )
    return 1.0 if result["correct"] else 0.0


# -- MIPROv2 Optimizer -------------------------------------------------------

class MIPROv2Optimizer:
    def __init__(self):
        self.retriever = WikipediaRetriever()
        self._setup_dspy()

    def _setup_dspy(self):
        """Configure DSPy to use LLMControls via the custom adapter."""
        configure_dspy(
            temperature=cfg.temperature,
            max_tokens=300,
        )
        logger.info(f"DSPy configured with {cfg.active_model} backend")

    def _make_program(self) -> dspy.Module:
        return dspy.ChainOfThought(BridgeQAWithReasoning)

    def _to_dspy_examples(
        self, samples: List[Dict[str, Any]]
    ) -> List[dspy.Example]:
        examples = []
        for i, s in enumerate(samples):
            logger.info(f"[MIPROv2 prep {i+1}/{len(samples)}] retrieving ...")
            docs = self.retriever.retrieve(s["question"])
            context = self.retriever.build_context(docs)
            ex = dspy.Example(
                question=s["question"],
                context=context,
                answer=s["answer"],
            ).with_inputs("question", "context")
            examples.append(ex)
        return examples

    def run(
        self,
        train_samples: List[Dict[str, Any]],
        val_samples: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        logger.info(f"MIPROv2 -- preparing {len(train_samples)} train examples ...")
        train_dspy = self._to_dspy_examples(train_samples)

        val_dspy = None
        if val_samples:
            logger.info(f"MIPROv2 -- preparing {len(val_samples)} val examples ...")
            val_dspy = self._to_dspy_examples(val_samples)

        program = self._make_program()

        logger.info("MIPROv2 -- starting teleprompter ...")
        try:
            teleprompter = dspy.MIPROv2(
                metric=hotpotqa_metric,
                auto=cfg.miprov2_auto,
                max_bootstrapped_demos=cfg.miprov2_max_bootstrapped,
                max_labeled_demos=cfg.miprov2_max_labeled,
                num_candidates=cfg.miprov2_num_candidates,
                verbose=True,
            )
            optimized = teleprompter.compile(
                program,
                trainset=train_dspy,
                valset=val_dspy,
            )
        except Exception as e:
            logger.warning(
                f"MIPROv2 full compile failed ({e}). "
                "Falling back to BootstrapFewShotWithRandomSearch."
            )
            teleprompter = dspy.BootstrapFewShotWithRandomSearch(
                metric=hotpotqa_metric,
                max_bootstrapped_demos=cfg.miprov2_max_bootstrapped,
                max_labeled_demos=cfg.miprov2_max_labeled,
                num_candidate_programs=cfg.miprov2_num_candidates,
            )
            optimized = teleprompter.compile(program, trainset=train_dspy)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = cfg.opt_dir / f"hotpotqa_miprov2_{timestamp}.json"
        optimized.save(str(out_path))
        logger.info(f"Optimised program saved -> {out_path}")

        val_metrics = {}
        summary_metrics = {}
        if val_dspy:
            val_metrics = self._evaluate_program(optimized, val_dspy)
            print_metrics(val_metrics)
            summary_metrics = {
                "yes_count": val_metrics["yes_count"],
                "no_count": val_metrics["no_count"],
                "accuracy": val_metrics["accuracy"],
                "accuracy_pct": val_metrics["accuracy_pct"],
                "wilson_ci95": val_metrics["wilson_ci95"],
            }

        return {
            "optimized_program_path": str(out_path),
            "summary_metrics": summary_metrics,
            "timestamp": timestamp,
        }

    @staticmethod
    def _evaluate_program(
        program, examples: List[dspy.Example]
    ) -> Dict[str, Any]:
        judge = LLMJudge()
        records = []
        for ex in examples:
            try:
                pred = program(question=ex.question, context=ex.context)
                result = judge.judge_sync(
                    question=ex.question,
                    gold_answer=ex.answer,
                    prediction=pred.answer,
                )
                records.append({
                    "judge_correct": result["correct"],
                })
            except Exception as e:
                logger.warning(f"Evaluation step failed: {e}")
                records.append({"judge_correct": False})
        return compute_metrics(records, split="val")

    def evaluate_test_set(
        self, program_path: str, test_samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Evaluate optimized program on full test set. Returns list of record dicts."""
        logger.info(f"Loading optimized program from {program_path} ...")
        program = self._make_program()
        program.load(program_path)

        logger.info(f"Preparing {len(test_samples)} test examples for MIPROv2 eval ...")
        test_dspy = self._to_dspy_examples(test_samples)

        judge = LLMJudge()
        records = []
        for i, ex in enumerate(test_dspy):
            try:
                pred = program(question=ex.question, context=ex.context)
                result = judge.judge_sync(
                    question=ex.question,
                    gold_answer=ex.answer,
                    prediction=pred.answer,
                )
                records.append({
                    "question": ex.question,
                    "prediction": pred.answer,
                    "gold_answer": ex.answer,
                    "correct": result["correct"],
                })
                logger.info(f"[MIPROv2 eval {i+1}/{len(test_dspy)}] correct={result['correct']}")
            except Exception as e:
                logger.warning(f"[MIPROv2 eval {i+1}] failed: {e}")
                records.append({
                    "question": ex.question,
                    "prediction": "UNKNOWN",
                    "gold_answer": ex.answer,
                    "correct": False,
                })

        # Save predictions JSONL so MIPROv2 results are recoverable
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        jsonl_path = cfg.output_dir / f"hotpotqa_miprov2_{timestamp}_predictions.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, default=str) + "\n")
        logger.info(f"MIPROv2 predictions saved -> {jsonl_path}")

        return records


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    loader = HotpotQALoader()
    train, val = loader.load_train_test(
        max_train=cfg.max_train_samples, max_test=50
    )
    optimizer = MIPROv2Optimizer()
    result = optimizer.run(train_samples=train, val_samples=val)
    print(f"\nMIPROv2 optimisation complete.")
    print(f"  Saved to : {result['optimized_program_path']}")
    if result["summary_metrics"]:
        m = result["summary_metrics"]
        print(f"  Val Accuracy : {m['accuracy_pct']}  CI95 {m['wilson_ci95']}")
        print(f"  Yes/No       : {m['yes_count']}/{m['no_count']}")


if __name__ == "__main__":
    main()
