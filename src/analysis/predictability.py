"""
Predictability analysis: train language-specific classifiers to predict
demographic attributes (gender, religion, nationality) from story text.

Higher classification accuracy = stronger demographic signal embedded in narratives.
Comparison across:
  - languages: cross-linguistic variation in signal strength
  - training stages (base vs RLHF): alignment's effect on demographic embedding
"""

import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"


def load_stories(stories_file: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Load stories and their demographic labels from a JSONL file."""
    texts, labels = [], {"child_gender": [], "religion": [], "parent_role": []}
    with open(stories_file, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["story"])
            for key in labels:
                labels[key].append(record[key])
    return texts, labels


def train_and_evaluate(
    texts: list[str],
    label_list: list[str],
    n_splits: int = 5,
) -> dict:
    """
    Train a TF-IDF + Logistic Regression classifier with cross-validation.
    Returns mean accuracy and full classification report for each fold.
    """
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    y = np.array(label_list)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_reports = []
    accuracies = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        report = classification_report(y[test_idx], preds, output_dict=True)
        fold_reports.append(report)
        accuracies.append(report["accuracy"])

    return {
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "fold_reports": fold_reports,
    }


def run_predictability_analysis(model_type: str) -> dict:
    """
    Run predictability analysis for all languages and demographic attributes.
    Returns a nested dict: results[language][demographic_attr] = metrics
    """
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    stories_root = Path(config["paths"]["stories"]) / f"{model_type}_model"
    results = {}

    for language in config["languages"]:
        stories_file = stories_root / language / "stories.jsonl"
        if not stories_file.exists():
            print(f"  No stories for {model_type}/{language}, skipping.")
            continue

        texts, labels = load_stories(stories_file)
        results[language] = {}

        for attr, label_list in labels.items():
            print(f"  Training classifier: {model_type}/{language}/{attr} ...")
            metrics = train_and_evaluate(texts, label_list)
            results[language][attr] = metrics
            print(f"    Mean accuracy: {metrics['mean_accuracy']:.3f} ± {metrics['std_accuracy']:.3f}")

    return results


def compare_base_vs_rlhf() -> dict:
    """Run analysis for both model types and return combined results for comparison."""
    results = {}
    for model_type in ["base", "rlhf"]:
        print(f"\n=== {model_type.upper()} model ===")
        results[model_type] = run_predictability_analysis(model_type)
    return results


if __name__ == "__main__":
    all_results = compare_base_vs_rlhf()
    # Print summary table
    print("\n=== Predictability Summary ===")
    for model_type, lang_results in all_results.items():
        for language, attr_results in lang_results.items():
            for attr, metrics in attr_results.items():
                print(
                    f"{model_type:6s} | {language:8s} | {attr:15s} | "
                    f"acc={metrics['mean_accuracy']:.3f} ± {metrics['std_accuracy']:.3f}"
                )
