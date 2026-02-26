# =============================================================================
# hotpotqa_loader.py  -- Official HotpotQA Dataset Loader
# Supports: local JSON, HuggingFace datasets, and demo fallback
# =============================================================================

import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

from scripts.config import cfg

logger = logging.getLogger(__name__)


class HotpotQALoader:
    """
    Loads HotpotQA from official JSON files or HuggingFace.

    Priority:
      1. Local hotpot_train_v1.1.json / hotpot_dev_distractor_v1.json
      2. HuggingFace hotpot_qa dataset
      3. Demo samples (smoke-test only)
    """

    def __init__(self):
        random.seed(cfg.random_seed)

    def load_train(self, max_samples: int = None) -> List[Dict[str, Any]]:
        limit = max_samples or cfg.max_train_samples
        return self._load_split(cfg.hotpot_train_file, "train", limit)

    def load_test(self, max_samples: int = None) -> List[Dict[str, Any]]:
        limit = max_samples or cfg.max_test_samples
        return self._load_split(cfg.hotpot_dev_file, "validation", limit)

    def load_train_test(
        self,
        max_train: int = None,
        max_test: int = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        train = self.load_train(max_train)
        test = self.load_test(max_test)
        logger.info(f"Dataset ready -- train: {len(train)} | test: {len(test)}")
        return train, test

    # -- Internal loaders ----------------------------------------------------

    def _load_split(self, local_file: str, hf_split: str, limit: int) -> List[Dict[str, Any]]:
        local_path = Path(local_file)
        if local_path.exists():
            return self._from_local_json(local_path, limit)
        try:
            return self._from_huggingface(hf_split, limit)
        except Exception as e:
            logger.warning(f"HuggingFace load failed: {e}")
        logger.warning("Using demo dataset -- results are NOT benchmark-valid.")
        return self._demo_samples()[:limit]

    def _from_local_json(self, path: Path, limit: int) -> List[Dict[str, Any]]:
        logger.info(f"Loading HotpotQA from {path} ...")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        random.shuffle(data)
        data = data[:limit]
        return [self._normalise(s) for s in data]

    def _from_huggingface(self, split: str, limit: int) -> List[Dict[str, Any]]:
        from datasets import load_dataset
        logger.info(f"Loading HotpotQA from HuggingFace (split={split}) ...")
        ds = load_dataset("hotpot_qa", "distractor", split=split)
        samples = []
        for i, item in enumerate(ds):
            if i >= limit:
                break
            samples.append(self._normalise(item))
        return samples

    @staticmethod
    def _normalise(item: Dict[str, Any]) -> Dict[str, Any]:
        sf_raw = item.get("supporting_facts", [])
        if isinstance(sf_raw, dict):
            titles = sf_raw.get("title", [])
            idxs = sf_raw.get("sent_id", [])
            supporting_facts = list(zip(titles, idxs))
        else:
            supporting_facts = [[f[0], f[1]] for f in sf_raw] if sf_raw else []

        ctx_raw = item.get("context", [])
        if isinstance(ctx_raw, dict):
            context = list(zip(ctx_raw.get("title", []), ctx_raw.get("sentences", [])))
        else:
            context = ctx_raw

        return {
            "_id": item.get("_id", item.get("id", f"demo_{id(item)}")),
            "question": item["question"],
            "answer": item["answer"],
            "supporting_facts": supporting_facts,
            "context": context,
            "type": item.get("type", "bridge"),
            "level": item.get("level", "medium"),
        }

    @staticmethod
    def _demo_samples() -> List[Dict[str, Any]]:
        return [
            {
                "_id": "demo_001",
                "question": "What is the headquarters city of the hotel chain owned by the Oberoi family?",
                "answer": "Delhi",
                "supporting_facts": [["EIH Limited", 0], ["Oberoi Hotels & Resorts", 1]],
                "context": [
                    ["EIH Limited", ["EIH Limited, branded as Oberoi Hotels & Resorts, is headquartered in Delhi, India."]],
                    ["Oberoi Hotels & Resorts", ["The Oberoi Hotels brand is operated by EIH Limited, a company founded by Mohan Singh Oberoi."]],
                ],
                "type": "bridge",
                "level": "easy",
            },
            {
                "_id": "demo_002",
                "question": "Cadmium chloride is soluble in what common solvent besides water?",
                "answer": "alcohol",
                "supporting_facts": [["Cadmium chloride", 0]],
                "context": [
                    ["Cadmium chloride", ["Cadmium chloride (CdCl2) is a white crystalline solid soluble in water, alcohol, and acetone."]],
                ],
                "type": "bridge",
                "level": "easy",
            },
            {
                "_id": "demo_003",
                "question": "Reliance Industries is headquartered in which city?",
                "answer": "Mumbai",
                "supporting_facts": [["Reliance Industries", 0]],
                "context": [
                    ["Reliance Industries", ["Reliance Industries Limited is an Indian conglomerate headquartered in Mumbai, Maharashtra, India."]],
                ],
                "type": "bridge",
                "level": "easy",
            },
        ]


def get_gold_context(sample: Dict[str, Any]) -> str:
    parts = []
    for title, sentences in sample.get("context", []):
        body = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
        parts.append(f"TITLE: {title}\n{body}")
    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = HotpotQALoader()
    train, test = loader.load_train_test(max_train=5, max_test=3)
    print(f"\nTrain[0]: {train[0]['question']!r}")
    print(f"  Answer : {train[0]['answer']!r}")
    print(f"  Type   : {train[0]['type']}")
    print(f"\nTest[0] : {test[0]['question']!r}")
    print(f"  SF titles: {[sf[0] for sf in test[0]['supporting_facts']]}")
