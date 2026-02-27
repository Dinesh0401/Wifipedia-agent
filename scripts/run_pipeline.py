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


def run_miprov2_eval(test_data, optimized_program_path):
    """Evaluate optimized MIPROv2 program on full test set (sync, runs in thread pool)."""
    from scripts.miprov2_pipeline import MIPROv2Optimizer

    logger.info("=" * 60)
    logger.info("STAGE 1b: MIPROv2 Test-Set Evaluation")
    logger.info("=" * 60)
    optimizer = MIPROv2Optimizer()
    verdicts = optimizer.evaluate_test_set(optimized_program_path, test_data)
    return verdicts


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
        print(f"Gold      : {r['task']['gold_answer']}")
        print(f"Pred      : {r['prediction']['answer']}")
        print(f"Judge     : verdict={r['metrics']['judge_verdict']} correct={r['metrics']['judge_correct']}")
        print(f"Reasoning : {r['metrics']['judge_reasoning']}")
        print()
    logger.info("Smoke test complete")


# -- Formatted TXT output ---------------------------------------------------

def _format_verdict_grid(verdicts, cols=5):
    """Format YES/NO verdicts in a grid, 5 per row."""
    lines = []
    for row_start in range(0, len(verdicts), cols):
        row_entries = []
        for j in range(row_start, min(row_start + cols, len(verdicts))):
            num = j + 1
            v = "YES" if verdicts[j] else "NO"
            entry = f"Q{num:03d}  {v}"
            row_entries.append(entry)
        padded = [e.ljust(12) for e in row_entries[:-1]] + [row_entries[-1]]
        lines.append("  " + "".join(padded))
    return "\n".join(lines)


def _format_per_question(records):
    """Format per-question details: Ques / Ans / Expected Ans / Correct."""
    lines = []
    for i, rec in enumerate(records):
        num = i + 1
        verdict = "YES" if rec["correct"] else "NO"
        lines.append(f"Q{num:03d}")
        lines.append(f"Ques:         {rec['question']}")
        lines.append(f"Ans:          {rec['prediction']}")
        lines.append(f"Expected Ans: {rec['gold_answer']}")
        lines.append(f"Correct:      {verdict}")
        lines.append("-" * 80)
    return "\n".join(lines)


def generate_results_txt(
    miprov2_records,
    ace_offline_records,
    ace_online_records,
    n_samples,
):
    """Generate the formatted pipeline results with per-question Ques/Ans/Expected Ans/Correct."""
    W = 80
    EQ = "=" * W
    HEAVY = "\u2501" * W  # ━

    def _stage_section(title, records):
        blk = []
        blk.append(HEAVY)
        blk.append(f"  {title}  (N={len(records)})")
        blk.append(HEAVY)
        blk.append("")
        blk.append(_format_per_question(records))
        blk.append("")
        yes = sum(1 for r in records if r["correct"])
        no = len(records) - yes
        acc = yes / len(records) * 100 if records else 0
        blk.append(f"  Accuracy: {acc:.1f}%  (YES: {yes}  NO: {no})")
        blk.append("")
        return "\n".join(blk)

    out = []
    out.append(EQ)
    out.append("  WIKIPEDIA QA AGENT \u2014 PIPELINE RESULTS")
    out.append("  Model  : claude-opus-4-6")
    out.append("  Dataset: HotpotQA (single-hop)")
    out.append("  Metric : LLM-as-Judge")
    out.append(f"  Samples: {n_samples}")
    out.append(EQ)
    out.append("")
    out.append("")

    out.append(_stage_section("STAGE 1 \u2014 MIPROv2", miprov2_records))
    out.append("")
    out.append(_stage_section("STAGE 2 \u2014 ACE OFFLINE", ace_offline_records))
    out.append("")
    out.append(_stage_section("STAGE 3 \u2014 ACE ONLINE", ace_online_records))
    out.append("")

    out.append(EQ)
    out.append("  OVERALL ACCURACY")
    out.append(EQ)
    out.append("")

    DASH = "\u2500" * 60  # ─
    out.append(f"  {'Stage':<16s}{'N':>3s}    {'YES':>4s}  {'NO':>4s}  {'Accuracy':>10s}")
    out.append(f"  {DASH}")

    for name, records in [
        ("MIPROv2", miprov2_records),
        ("ACE Offline", ace_offline_records),
        ("ACE Online", ace_online_records),
    ]:
        yes = sum(1 for r in records if r["correct"])
        no = len(records) - yes
        acc = f"{yes / len(records) * 100:.1f}%" if records else "0.0%"
        out.append(
            f"  {name:<16s}{len(records):>3d}    {yes:>4d}  {no:>4d}   {acc:>8s}"
        )

    out.append(f"  {DASH}")
    out.append("")
    out.append(EQ)

    return "\n".join(out)


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
    miprov2_records = []

    # MIPROv2 — run in thread pool to avoid blocking the event loop
    if args.task in ("miprov2", "all") and not args.no_mipro:
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            results["miprov2"] = await loop.run_in_executor(
                pool, run_miprov2, train_data, val_data
            )
        optimized_program_path = results["miprov2"].get("optimized_program_path")

        # Evaluate MIPROv2 optimized program on full test set
        if optimized_program_path:
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                miprov2_records = await loop.run_in_executor(
                    pool, run_miprov2_eval, test_data, optimized_program_path
                )

    # ACE Offline — uses optimized program if MIPROv2 ran
    if args.task in ("ace", "all") and args.mode in ("offline", "both"):
        results["ace_offline"] = await run_ace_offline(test_data, optimized_program_path)

    # ACE Online — uses optimized program if MIPROv2 ran
    if args.task in ("ace", "all") and args.mode in ("online", "both"):
        results["ace_online"] = await run_ace_online(test_data, optimized_program_path)

    # -- Collect per-sample records ----
    ace_offline_records = results.get("ace_offline", {}).get("per_sample_records", [])
    ace_online_records = results.get("ace_online", {}).get("per_sample_records", [])

    n_samples = len(test_data)

    # -- Generate formatted TXT output ----
    if miprov2_records and ace_offline_records and ace_online_records:
        txt_output = generate_results_txt(
            miprov2_records=miprov2_records,
            ace_offline_records=ace_offline_records,
            ace_online_records=ace_online_records,
            n_samples=n_samples,
        )

        # Save to results/ directory
        results_dir = Path(cfg.output_dir).parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = results_dir / f"pipeline_results_singlehop_{timestamp}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_output)

        print("\n" + txt_output)
        print(f"\nResults saved to: {txt_path}")
    else:
        # Fallback summary for partial runs
        print("\n" + "=" * 70)
        print("  FINAL RESULTS SUMMARY")
        print("=" * 70)
        for name, res in results.items():
            if "summary_metrics" in res and res["summary_metrics"]:
                m = res["summary_metrics"]
                print(f"\n  [{name.upper()}]")
                print(f"    Accuracy : {m['accuracy_pct']}")
                print(f"    Yes/No   : {m.get('yes_count', '-')}/{m.get('no_count', '-')}")
                if "ace_faithfulness_pct" in m:
                    print(f"    Faithful : {m['ace_faithfulness_pct']}")
        print("=" * 70)

    print(f"\nAll outputs saved in: {cfg.output_dir}/")


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
