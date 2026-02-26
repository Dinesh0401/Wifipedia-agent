# =============================================================================
# metrics.py  -- Research-Grade Evaluation Metrics (FINER-ORD Format)
# Precision + Recall + F1 with Bootstrap CIs
# Based on official HotpotQA evaluation script
# =============================================================================

import re
import string
import random
import numpy as np
from typing import List, Tuple, Dict, Any


# -- Answer normalisation ---------------------------------------------------

def normalize_answer(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b\s*", " ", text)
    return " ".join(text.split())


# -- Core metrics -----------------------------------------------------------

def token_precision_recall_f1(pred: str, gold: str) -> Tuple[float, float, float]:
    """Compute token-level precision, recall, and F1 between pred and gold."""
    p_tokens = normalize_answer(pred).split()
    g_tokens = normalize_answer(gold).split()
    if not p_tokens and not g_tokens:
        return 1.0, 1.0, 1.0
    if not p_tokens or not g_tokens:
        return 0.0, 0.0, 0.0
    common = set(p_tokens) & set(g_tokens)
    if not common:
        return 0.0, 0.0, 0.0
    precision = len(common) / len(p_tokens)
    recall = len(common) / len(g_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def f1_score(pred: str, gold: str) -> float:
    """Backward-compatible F1 score (used by DSPy metric)."""
    _, _, f1 = token_precision_recall_f1(pred, gold)
    return f1


def exact_match(pred: str, gold: str) -> bool:
    """Backward-compatible exact match (used by DSPy metric)."""
    return normalize_answer(pred) == normalize_answer(gold)


def supporting_fact_f1(
    pred_titles: List[str],
    gold_facts: List[List],
) -> float:
    gold_titles = set()
    for fact in gold_facts:
        if isinstance(fact, (list, tuple)) and len(fact) >= 1:
            gold_titles.add(str(fact[0]))
    pred_titles_set = set(str(t) for t in pred_titles)
    if not gold_titles:
        return 0.0
    common = gold_titles & pred_titles_set
    precision = len(common) / len(pred_titles_set) if pred_titles_set else 0.0
    recall = len(common) / len(gold_titles)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# -- Statistical confidence intervals --------------------------------------

def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    random.seed(seed)
    np.random.seed(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    boot_means = [
        np.mean(random.choices(values, k=n))
        for _ in range(n_bootstrap)
    ]
    alpha = 1 - ci
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# -- Aggregate summary (FINER-ORD format) ----------------------------------

def compute_metrics(records: List[Dict[str, Any]], split: str = "test") -> Dict[str, Any]:
    n = len(records)
    prec_vals = [float(r["precision"]) for r in records]
    rec_vals = [float(r["recall"]) for r in records]
    f1_vals = [float(r["f1"]) for r in records]

    prec_mean = float(np.mean(prec_vals))
    rec_mean = float(np.mean(rec_vals))
    f1_mean = float(np.mean(f1_vals))

    prec_ci_lo, prec_ci_hi = bootstrap_ci(prec_vals)
    rec_ci_lo, rec_ci_hi = bootstrap_ci(rec_vals)
    f1_ci_lo, f1_ci_hi = bootstrap_ci(f1_vals)

    return {
        "split": split,
        "n": n,
        "precision_mean": round(prec_mean, 4),
        "precision_min": round(float(min(prec_vals)), 4) if prec_vals else 0.0,
        "precision_max": round(float(max(prec_vals)), 4) if prec_vals else 0.0,
        "precision_ci95": {
            "lower": round(prec_ci_lo, 4),
            "upper": round(prec_ci_hi, 4),
            "pct": f"[{prec_ci_lo*100:.1f}%, {prec_ci_hi*100:.1f}%]",
        },
        "recall_mean": round(rec_mean, 4),
        "recall_min": round(float(min(rec_vals)), 4) if rec_vals else 0.0,
        "recall_max": round(float(max(rec_vals)), 4) if rec_vals else 0.0,
        "recall_ci95": {
            "lower": round(rec_ci_lo, 4),
            "upper": round(rec_ci_hi, 4),
            "pct": f"[{rec_ci_lo*100:.1f}%, {rec_ci_hi*100:.1f}%]",
        },
        "f1_mean": round(f1_mean, 4),
        "f1_min": round(float(min(f1_vals)), 4) if f1_vals else 0.0,
        "f1_max": round(float(max(f1_vals)), 4) if f1_vals else 0.0,
        "f1_ci95": {
            "lower": round(f1_ci_lo, 4),
            "upper": round(f1_ci_hi, 4),
            "pct": f"[{f1_ci_lo*100:.1f}%, {f1_ci_hi*100:.1f}%]",
        },
    }


def print_metrics(m: Dict[str, Any]) -> None:
    print(f"\n{'='*60}")
    print(f"  Results -- {m['split'].upper()} SET  (N={m['n']})")
    print(f"{'='*60}")
    print(f"  Precision   : {m['precision_mean']*100:.1f}%  CI95 {m['precision_ci95']['pct']}")
    print(f"  Recall      : {m['recall_mean']*100:.1f}%  CI95 {m['recall_ci95']['pct']}")
    print(f"  F1          : {m['f1_mean']*100:.1f}%  CI95 {m['f1_ci95']['pct']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    assert exact_match("Delhi", "delhi") is True
    assert exact_match("New Delhi", "Delhi") is False
    p, r, f = token_precision_recall_f1("the cat sat", "the dog sat")
    assert abs(f - 0.5) < 1e-6
    p, r, f = token_precision_recall_f1("cat", "cat")
    assert abs(f - 1.0) < 1e-6
    assert abs(p - 1.0) < 1e-6
    assert abs(r - 1.0) < 1e-6
    print("All metric unit tests passed")
