"""Train the deployed language and topic models on ALL labelled data and save them to models/.

(scripts/evaluate.py trains separate copies on the train split only, to measure them on test.)

Run:  python scripts/train_models.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.ingest import build_chunks, extract_pages, save_chunks  # noqa: E402
from src.language import MLLanguageDetector  # noqa: E402
from src.ingest import load_chunks  # noqa: E402
from src.topic_classifier import build_topic_model, weak_labels_from_policy  # noqa: E402


def main():
    if not config.CHUNKS_FILE.exists():
        save_chunks(build_chunks(extract_pages()))
    qa = pd.read_csv(config.QA_FILE)
    clf = pd.read_csv(config.CLASSIFIER_FILE)

    lang = MLLanguageDetector().fit(qa.input, qa.language)
    joblib.dump(lang, config.LANG_MODEL_FILE)
    print(f"language model: {len(qa)} questions -> {config.LANG_MODEL_FILE}")

    weak = weak_labels_from_policy(load_chunks())
    X = pd.concat([clf[["text", "topic"]], weak], ignore_index=True)
    topic = build_topic_model().fit(X.text, X.topic)
    joblib.dump(topic, config.TOPIC_MODEL_FILE)
    print(f"topic model: {len(clf)} labelled + {len(weak)} weakly-labelled policy chunks, "
          f"{X.topic.nunique()} topics -> {config.TOPIC_MODEL_FILE}")


if __name__ == "__main__":
    main()
