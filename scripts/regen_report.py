"""Regenerate pipeline_results.txt from JSONL prediction files."""
import json

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

mipro_raw = load_jsonl("research_outputs/hotpotqa_miprov2_20260227_220851_predictions.jsonl")
offline_raw = load_jsonl("research_outputs/hotpotqa_ace_offline_20260227_200546_predictions.jsonl")
online_raw = load_jsonl("research_outputs/hotpotqa_ace_online_20260227_204514_predictions.jsonl")


def normalise(record):
    """Return a flat dict {question, prediction, gold_answer, correct} regardless of source format."""
    # MIPROv2 format: flat keys
    if "question" in record and "prediction" in record:
        return record
    # ACE format: nested keys
    return {
        "question": record["task"]["question"],
        "prediction": record["prediction"]["answer"],
        "gold_answer": record["task"]["gold_answer"],
        "correct": record["metrics"]["judge_correct"],
    }


mipro = [normalise(r) for r in mipro_raw]
offline = [normalise(r) for r in offline_raw]
online = [normalise(r) for r in online_raw]


def fmt(records, stage_name):
    lines = []
    bar = "\u2501" * 80
    thin = "-" * 80
    lines.append(bar)
    lines.append(f"  {stage_name}  (N={len(records)})")
    lines.append(bar)
    lines.append("")
    yes = 0
    for i, r in enumerate(records):
        qid = f"Q{i+1:03d}"
        corr = "YES" if r["correct"] else "NO"
        if r["correct"]:
            yes += 1
        lines.append(qid)
        lines.append(f"Ques:         {r['question']}")
        lines.append(f"Ans:          {r['prediction']}")
        lines.append(f"Expected Ans: {r['gold_answer']}")
        lines.append(f"Correct:      {corr}")
        lines.append(thin)
    acc = yes / len(records) * 100 if records else 0
    lines.append("")
    lines.append(f"Overall Accuracy: {yes}/{len(records)} = {acc:.1f}%")
    lines.append("")
    return "\n".join(lines)

header = (
    "=" * 80 + "\n"
    "  WIKIPEDIA QA AGENT \u2014 PIPELINE RESULTS\n"
    "  Model  : claude-opus-4-6\n"
    "  Dataset: HotpotQA (single-hop)\n"
    "  Metric : LLM-as-Judge\n"
    "  Samples: 100\n"
    + "=" * 80 + "\n\n"
)

s1 = fmt(mipro, "STAGE 1 \u2014 MIPROv2")
s2 = fmt(offline, "STAGE 2 \u2014 ACE Offline")
s3 = fmt(online, "STAGE 3 \u2014 ACE Online")

full = header + "\n" + s1 + "\n\n" + s2 + "\n\n" + s3

for path in [
    "research_outputs/pipeline_results.txt",
    "results/pipeline_results_singlehop_20260227_204514.txt",
]:
    with open(path, "w", encoding="utf-8") as f:
        f.write(full)
    print(f"Wrote {path} ({len(full)} chars)")

print("Done")
