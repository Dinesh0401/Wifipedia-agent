# =============================================================================
# metrics.py  -- Research-Grade Evaluation Metrics
# EM + F1 + Wilson CI (EM) + Bootstrap CI (F1)
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

def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def f1_score(pred: str, gold: str) -> float:
    p_tokens = normalize_answer(pred).split()
    g_tokens = normalize_answer(gold).split()
    if not p_tokens or not g_tokens:
        return 0.0
    common = set(p_tokens) & set(g_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(p_tokens)
    recall = len(common) / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


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

def wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


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


# -- Aggregate summary ------------------------------------------------------

def compute_metrics(records: List[Dict[str, Any]], split: str = "test") -> Dict[str, Any]:
    n = len(records)
    em_vals = [float(r["exact_match"]) for r in records]
    f1_vals = [float(r["f1_score"]) for r in records]

    em_mean = float(np.mean(em_vals))
    f1_mean = float(np.mean(f1_vals))
    f1_std = float(np.std(f1_vals))

    em_ci_lo, em_ci_hi = wilson_ci(em_mean, n)
    f1_ci_lo, f1_ci_hi = bootstrap_ci(f1_vals)

    return {
        "split": split,
        "n": n,
        "exact_match": {
            "mean": round(em_mean, 4),
            "pct": f"{em_mean * 100:.1f}%",
            "wilson_ci95": {
                "lower": round(em_ci_lo, 4),
                "upper": round(em_ci_hi, 4),
                "pct": f"[{em_ci_lo*100:.1f}%, {em_ci_hi*100:.1f}%]",
            },
        },
        "f1_score": {
            "mean": round(f1_mean, 4),
            "std": round(f1_std, 4),
            "pct": f"{f1_mean * 100:.1f}%",
            "bootstrap_ci95": {
                "lower": round(f1_ci_lo, 4),
                "upper": round(f1_ci_hi, 4),
                "pct": f"[{f1_ci_lo*100:.1f}%, {f1_ci_hi*100:.1f}%]",
            },
        },
    }


def print_metrics(m: Dict[str, Any]) -> None:
    print(f"\n{'='*60}")
    print(f"  Results -- {m['split'].upper()} SET  (N={m['n']})")
    print(f"{'='*60}")
    em = m["exact_match"]
    f1 = m["f1_score"]
    print(f"  Exact Match  : {em['pct']}  CI95 {em['wilson_ci95']['pct']}")
    print(f"  F1 Score     : {f1['pct']}  CI95 {f1['bootstrap_ci95']['pct']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    assert exact_match("Delhi", "delhi") is True
    assert exact_match("New Delhi", "Delhi") is False
    # After normalization (articles removed): "cat sat" vs "dog sat" -> F1 = 0.5
    assert abs(f1_score("the cat sat", "the dog sat") - 0.5) < 1e-6
    assert abs(f1_score("cat", "cat") - 1.0) < 1e-6
    print("All metric unit tests passed")
