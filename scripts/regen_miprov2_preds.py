"""
One-time script: Re-run MIPROv2 eval to capture actual predictions.
After this, regenerate the pipeline_results.txt with real answers.
"""
import json
import logging
from scripts.config import cfg
from scripts.hotpotqa_loader import HotpotQALoader
from scripts.miprov2_pipeline import MIPROv2Optimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Load same test data
loader = HotpotQALoader()
_, test_data = loader.load_train_test(max_train=cfg.max_train_samples, max_test=cfg.max_test_samples)
print(f"Test samples: {len(test_data)}")

# The latest optimized program from the pipeline run
program_path = "optimized_programs/hotpotqa_miprov2_20260227_191204.json"
print(f"Using program: {program_path}")

# Run eval - saves JSONL and returns records with predictions
optimizer = MIPROv2Optimizer()
records = optimizer.evaluate_test_set(program_path, test_data)

yes_count = sum(1 for r in records if r["correct"])
print(f"\nDone: {yes_count} YES / {len(records)-yes_count} NO out of {len(records)}")
print(f"Sample record:\n{json.dumps(records[0], indent=2)}")
