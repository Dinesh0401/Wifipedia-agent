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
import numpy as np

from scripts.metrics import exact_match, f1_score, compute_metrics, print_metrics
from scripts.hotpotqa_loader import HotpotQALoader
from scripts.wiki_retriever import WikipediaRetriever
from scripts.dspy_adapter import configure_dspy
from scripts.config import cfg

logger = logging.getLogger(__name__)


# -- DSPy signatures --------------------------------------------------------

class BridgeQA(dspy.Signature):
    """Answer a multi-hop Wikipedia bridge question given retrieved context."""
    question = dspy.InputField(desc="A multi-hop factual question")
    context = dspy.InputField(desc="Retrieved Wikipedia passages (may contain distractors)")
    answer = dspy.OutputField(desc="Short factual answer (1-5 words)")


class BridgeQAWithReasoning(dspy.Signature):
    """Solve a multi-hop Wikipedia bridge question with chain-of-thought."""
    question = dspy.InputField(desc="Multi-hop factual question")
    context = dspy.InputField(desc="Retrieved Wikipedia passages")
    reasoning = dspy.OutputField(desc="Chain-of-thought: Entity1 -> bridge -> Entity2")
    answer = dspy.OutputField(desc="Concise final answer (1-5 words)")


# -- HotpotQA metric for DSPy -----------------------------------------------

def hotpotqa_metric(example: dspy.Example, pred, trace=None) -> float:
    gold = example.answer
    pred_ans = getattr(pred, "answer", "") or ""
    return f1_score(pred_ans, gold)


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
        if val_dspy:
            val_metrics = self._evaluate_program(optimized, val_dspy)
            print_metrics(val_metrics)

        return {
            "optimized_program_path": str(out_path),
            "val_metrics": val_metrics,
            "timestamp": timestamp,
        }

    @staticmethod
    def _evaluate_program(
        program, examples: List[dspy.Example]
    ) -> Dict[str, Any]:
        records = []
        for ex in examples:
            try:
                pred = program(question=ex.question, context=ex.context)
                em = exact_match(pred.answer, ex.answer)
                f1 = f1_score(pred.answer, ex.answer)
                records.append({"exact_match": em, "f1_score": f1})
            except Exception as e:
                logger.warning(f"Evaluation step failed: {e}")
                records.append({"exact_match": False, "f1_score": 0.0})
        return compute_metrics(records, split="val")


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
    if result["val_metrics"]:
        em = result["val_metrics"]["exact_match"]["pct"]
        f1 = result["val_metrics"]["f1_score"]["pct"]
        print(f"  Val EM   : {em}")
        print(f"  Val F1   : {f1}")


if __name__ == "__main__":
    main()
