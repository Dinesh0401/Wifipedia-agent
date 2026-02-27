# =============================================================================
# metrics.py  -- LLM-as-a-Judge Aggregate Metrics (Yes/No)
# Wilson CI for binomial confidence intervals
# Accuracy = count(Yes) / total
# =============================================================================

import math
from typing import List, Dict, Any, Tuple


# -- Wilson score confidence interval -----------------------------------------

def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% CI by default)."""
    if n == 0:
        return 0.0, 0.0
    denominator = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denominator
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denominator
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return round(lo, 4), round(hi, 4)


# -- Aggregate summary (LLM-as-a-Judge Yes/No) --------------------------------

def compute_metrics(records: List[Dict[str, Any]], split: str = "test") -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"split": split, "n": 0}

    yes_count = sum(1 for r in records if bool(r["judge_correct"]))
    accuracy = yes_count / n
    lo, hi = wilson_ci(accuracy, n)

    return {
        "split": split,
        "n": n,
        "yes_count": yes_count,
        "no_count": n - yes_count,
        "accuracy": round(accuracy, 4),
        "accuracy_pct": f"{accuracy * 100:.1f}%",
        "wilson_ci95": f"[{lo * 100:.1f}%, {hi * 100:.1f}%]",
    }


def print_metrics(m: Dict[str, Any]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Results -- {m['split'].upper()} SET  (N={m['n']})")
    print(f"{'=' * 60}")
    print(f"  Yes : {m.get('yes_count', 0)}  |  No : {m.get('no_count', 0)}")
    print(f"  Accuracy : {m['accuracy_pct']}  CI95 {m['wilson_ci95']}")
    print(f"{'=' * 60}\n")
