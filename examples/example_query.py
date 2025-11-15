"""Example that demonstrates classifying entries from HarmActEval_dataset.json.

This script will try to load the real `ActionClassifier`. If the heavy deps or model
are not available, it falls back to a lightweight demo classifier so you can see
the expected outputs and usage without installing torch / sentence-transformers.
"""

import json
from pathlib import Path
from typing import Tuple

from agent_action_classifier import ActionClassifier


class DemoClassifier:
    """Simple deterministic fallback classifier for demo runs.

    It inspects the `label` and `resource` fields and marks known-harmful labels
    as harmful. This is used only when the real model can't be loaded.
    """

    HARMFUL_LABELS = {
        "harassment",
        "misinformation_spread",
        "harmful_content_generation",
        "deepfake_manipulation",
        "child_exploitation",
        "extremist_propaganda",
    }

    def predict(self, action_dict) -> Tuple[bool, float]:
        label = action_dict.get("label", "")
        resource = action_dict.get("resource", "")
        if label in self.HARMFUL_LABELS or any(k in resource for k in ["cyberbullying", "misinformation", "illegal"]):
            return True, 0.95
        return False, 0.85


def load_samples() -> list:
    p = Path(__file__).parent.parent / "action_classifier" / "HarmActEval_dataset.json"
    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("actions", [])


def main():
    samples = load_samples()

    # Try to create the real classifier; if that fails, use the DemoClassifier
    try:
        clf = ActionClassifier()
        print("Loaded real ActionClassifier (embedding + model).")
    except Exception as e:  # pragma: no cover - demo fallback
        print("Could not load real model/embeddings. Falling back to DemoClassifier.")
        print("Reason:", str(e))
        clf = DemoClassifier()

    # Classify a small subset (first 10) so output is short
    for entry in samples[:10]:
        # The training data stores label/resource at the top level and action in `action` key
        is_harmful, conf = clf.predict(entry)
        print(f"action_id={entry.get('action_id')}, label={entry.get('label')}, resource={entry.get('resource')}")
        print(f" -> harmful={is_harmful}, confidence={conf:.2f}\n")


if __name__ == "__main__":
    main()
