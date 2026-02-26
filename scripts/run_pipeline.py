#!/usr/bin/env python3
# =============================================================================
# run_pipeline.py  -- CLI Runner for Wiki Agent + MIPROv2 + ACE
#
# Usage:
#   python -m scripts.run_pipeline --task smoke
#   python -m scripts.run_pipeline --task miprov2
#   python -m scripts.run_pipeline --task ace --mode offline
#   python -m scripts.run_pipeline --task ace --mode online
#   python -m scripts.run_pipeline --task all
#
#   uv run python -m scripts.run_pipeline --task all
# =============================================================================

import argparse
import asyncio
import concurrent.futures
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from scripts.config import cfg, VALID_MODELS
from scripts.hotpotqa_loader import HotpotQALoader
from scripts.metrics import print_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Wikipedia Agent Research Pipeline -- MIPROv2 + ACE"
    )
    p.add_argument(
        "--task",
        choices=["miprov2", "ace", "all", "smoke"],
        default="all",
        help="Which pipeline stage to run",
    )
    p.add_argument(
        "--mode",
        choices=["offline", "online", "both"],
        default="both",
        help="ACE mode (only relevant when --task ace or --task all)",
    )
    p.add_argument("--train", type=int, default=None, help="Max train samples")
    p.add_argument("--test", type=int, default=None, help="Max test samples")
    p.add_argument(
        "--model",
        choices=list(VALID_MODELS),
        default=None,
        help="LLM backend to use (overrides ACTIVE_MODEL env var)",
    )
    p.add_argument("--no-mipro", action="store_true", help="Skip MIPROv2")
    return p.parse_args()


# -- Stage 1: MIPROv2 -------------------------------------------------------

def run_miprov2(train_data, val_data):
    from scripts.miprov2_pipeline import MIPROv2Optimizer

    logger.info("=" * 60)
    logger.info("STAGE 1: MIPROv2 Prompt Optimisation")
    logger.info("=" * 60)
    optimizer = MIPROv2Optimizer()
    result = optimizer.run(train_data, val_samples=val_data)
    logger.info(f"MIPROv2 -> {result['optimized_program_path']}")
    return result


# -- Stage 2: ACE Offline ---------------------------------------------------

async def run_ace_offline(test_data, optimized_program_path=None):
    from scripts.ace_pipeline import ACERunner

    logger.info("=" * 60)
    logger.info("STAGE 2: ACE Offline Evaluation")
    logger.info("=" * 60)
    runner = ACERunner(online_mode=False, optimized_program_path=optimized_program_path)
    result = await runner.run_offline(test_data)
    return result


# -- Stage 3: ACE Online ----------------------------------------------------

async def run_ace_online(test_data, optimized_program_path=None):
    from scripts.ace_pipeline import ACERunner

    logger.info("=" * 60)
    logger.info("STAGE 3: ACE Online Evaluation (with Reflection)")
    logger.info("=" * 60)
    runner = ACERunner(online_mode=True, optimized_program_path=optimized_program_path)
    result = await runner.run_online(test_data)
    return result


# -- Smoke test --------------------------------------------------------------

async def run_smoke():
    from scripts.wiki_agent import WikiAgent

    loader = HotpotQALoader()
    _, test = loader.load_train_test(max_train=1, max_test=2)

    agent = WikiAgent()
    logger.info("Running smoke test on 2 samples ...")
    results = await agent.run_batch(test, concurrency=1)

    for r in results:
        print(f"Q: {r['task']['question']}")
        print(f"Gold  : {r['task']['gold_answer']}")
        print(f"Pred  : {r['prediction']['answer']}")
        print(f"EM    : {r['metrics']['exact_match']}")
        print(f"F1    : {r['metrics']['f1_score']:.2f}")
        print()
    logger.info("Smoke test complete")


# -- Main --------------------------------------------------------------------

async def main_async(args):
    # Override active model if --model flag is provided
    if args.model:
        cfg.active_model = args.model
        logger.info(f"Model backend set to: {cfg.active_model}")
    logger.info(f"Using model: {cfg.get_display_model_name()}")

    if args.task == "smoke":
        await run_smoke()
        return

    loader = HotpotQALoader()
    max_tr = args.train or cfg.max_train_samples
    max_te = args.test or cfg.max_test_samples

    train_data, test_data = loader.load_train_test(max_train=max_tr, max_test=max_te)
    val_data = test_data[: min(20, len(test_data))]

    results = {}
    optimized_program_path = None

    # MIPROv2 — run in thread pool to avoid blocking the event loop
    if args.task in ("miprov2", "all") and not args.no_mipro:
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            results["miprov2"] = await loop.run_in_executor(
                pool, run_miprov2, train_data, val_data
            )
        optimized_program_path = results["miprov2"].get("optimized_program_path")

    # ACE Offline — uses optimized program if MIPROv2 ran
    if args.task in ("ace", "all") and args.mode in ("offline", "both"):
        results["ace_offline"] = await run_ace_offline(test_data, optimized_program_path)

    # ACE Online — uses optimized program if MIPROv2 ran
    if args.task in ("ace", "all") and args.mode in ("online", "both"):
        results["ace_online"] = await run_ace_online(test_data, optimized_program_path)

    # -- Final comparison ----
    print("\n" + "=" * 70)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 70)

    for name, res in results.items():
        if "metrics" in res:
            m = res["metrics"]
            print(f"\n  [{name.upper()}]")
            print(
                f"    EM        : {m['exact_match']['pct']}  "
                f"CI95 {m['exact_match']['wilson_ci95']['pct']}"
            )
            print(
                f"    F1        : {m['f1_score']['pct']}  "
                f"CI95 {m['f1_score']['bootstrap_ci95']['pct']}"
            )
            if "ace_faithfulness" in m:
                print(f"    Faithful  : {m['ace_faithfulness']['pct']}")

    print("=" * 70)
    print(f"\nAll outputs saved in: {cfg.output_dir}/")


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
