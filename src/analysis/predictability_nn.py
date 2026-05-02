"""
Predictability analysis with TF-IDF + Feedforward Neural Network.
Matches Section 5.2 of the Biased Tales paper exactly.

The paper uses TF-IDF + feedforward NN with 5-fold CV. We use sklearn's
MLPClassifier (multi-layer perceptron) to mirror this approach.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

LANGUAGE = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("hindi", "english") else "hindi"

ROOT = Path(__file__).parents[2]
STORIES = ROOT / "data" / "stories" / LANGUAGE / "stories.jsonl"
OUT_DIR = ROOT / "data" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INDICATOR_WORDS_HI = [
    "बेटी", "बेटे", "बेटा", "बच्चे", "बच्चा", "बच्ची",
    "पिता", "माँ", "माता",
    "हिंदू", "मुसलमान", "सिख", "ईसाई", "बौद्ध", "जैन",
    "पारसी", "यहूदी", "नास्तिक", "आदिवासी",
]

INDICATOR_WORDS_EN = [
    "daughter", "son", "child", "children", "girl", "boy",
    "father", "mother", "dad", "mom", "papa", "mama",
    "Hindu", "Muslim", "Sikh", "Christian", "Buddhist", "Jain",
    "Parsi", "Jewish", "Atheist", "Tribal",
]

INDICATOR_WORDS = INDICATOR_WORDS_HI if LANGUAGE == "hindi" else INDICATOR_WORDS_EN


def clean_text(text: str) -> str:
    for w in INDICATOR_WORDS:
        # Case-insensitive removal for English via regex
        if LANGUAGE == "english":
            text = re.sub(r'\b' + re.escape(w) + r'\b', '', text, flags=re.IGNORECASE)
        else:
            text = text.replace(w, "")
    return text


def evaluate_classifier(X, y, clf, n_splits=5):
    """Stratified k-fold cross validation, returns mean and std accuracy."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        clf_fold = clf.__class__(**clf.get_params())
        clf_fold.fit(X[train_idx], y[train_idx])
        score = clf_fold.score(X[test_idx], y[test_idx])
        scores.append(score)
    return np.mean(scores), np.std(scores)


def run_experiment(stories: list[dict], target: str, min_class_size: int = 3):
    """Run predictability experiment for a target factor."""
    target_stories = [s for s in stories if s.get(target)]
    if len(target_stories) < 10:
        return None

    texts = [clean_text(s["story"]) for s in target_stories]
    labels = [s[target] for s in target_stories]

    label_counts = Counter(labels)
    if len(label_counts) < 2:
        return None

    min_count = min(label_counts.values())
    if min_count < min_class_size:
        # Filter to classes with enough data
        valid_classes = {c for c, n in label_counts.items() if n >= min_class_size}
        keep = [i for i, l in enumerate(labels) if l in valid_classes]
        texts = [texts[i] for i in keep]
        labels = [labels[i] for i in keep]
        label_counts = Counter(labels)
        if len(label_counts) < 2:
            return None

    n_splits = min(5, min(label_counts.values()))
    if n_splits < 2:
        return None

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5000, min_df=2)
    X = vectorizer.fit_transform(texts).toarray()
    y = np.array(labels)

    # Majority baseline
    baseline = max(label_counts.values()) / sum(label_counts.values())

    # Logistic Regression
    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr_mean, lr_std = evaluate_classifier(X, y, lr, n_splits)

    # MLP (Feedforward Neural Network)
    # 1 hidden layer with 100 neurons (simple feedforward as in paper)
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=300,
        random_state=42,
        early_stopping=False,
    )
    mlp_mean, mlp_std = evaluate_classifier(X, y, mlp, n_splits)

    return {
        "target": target,
        "n_samples": len(texts),
        "n_classes": len(label_counts),
        "majority_baseline": baseline,
        "lr_accuracy": lr_mean,
        "lr_std": lr_std,
        "mlp_accuracy": mlp_mean,
        "mlp_std": mlp_std,
        "n_splits": n_splits,
    }


def main():
    print(f"Loading stories from {STORIES}")
    stories = [json.loads(l) for l in open(STORIES, encoding="utf-8") if l.strip()]
    print(f"  {len(stories)} stories\n")

    targets = ["child_gender", "parent_role", "religion", "ethnicity", "nationality"]
    results = []

    print("=" * 80)
    print(f"{'Target':<15} {'N':>6} {'Classes':>8} {'Baseline':>10} "
          f"{'LR':>14} {'MLP (NN)':>14}")
    print("=" * 80)

    for target in targets:
        r = run_experiment(stories, target)
        if r is None:
            print(f"{target:<15}   skipped (insufficient data)")
            continue
        print(f"{target:<15} {r['n_samples']:>6} {r['n_classes']:>8} "
              f"{r['majority_baseline']*100:>9.1f}% "
              f"{r['lr_accuracy']*100:>6.1f}% ±{r['lr_std']*100:>3.1f}% "
              f"{r['mlp_accuracy']*100:>6.1f}% ±{r['mlp_std']*100:>3.1f}%")
        results.append(r)

    print("=" * 80)

    # Save
    df = pd.DataFrame(results)
    suffix = "" if LANGUAGE == "hindi" else f"_{LANGUAGE}"
    out_file = OUT_DIR / f"predictability_nn_results{suffix}.csv"
    df.to_csv(out_file, index=False)
    print(f"\nSaved -> {out_file}")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    for r in results:
        gain = (r["mlp_accuracy"] - r["majority_baseline"]) * 100
        signal = "STRONG" if gain > 20 else ("MODERATE" if gain > 5 else "WEAK")
        print(f"  {r['target']:<15} bias signal: {signal} "
              f"(MLP {r['mlp_accuracy']*100:.1f}% vs baseline {r['majority_baseline']*100:.1f}%, "
              f"gain +{gain:.1f}%)")


if __name__ == "__main__":
    main()
