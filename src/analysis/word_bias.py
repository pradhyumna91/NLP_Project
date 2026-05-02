"""
Surface-level word bias analysis (Section 5.1 of Biased Tales paper).
Computes Pearson correlation between word presence and sociocultural factors.
For each (factor_value, word) pair, the correlation indicates how strongly
the word appears in stories with that factor value vs others.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

LANGUAGE = sys.argv[1] if len(sys.argv) > 1 else "hindi"

ROOT = Path(__file__).parents[2]
STORIES = ROOT / "data" / "stories" / LANGUAGE / "stories.jsonl"
OUT_DIR = ROOT / "data" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Words to exclude from analysis (obvious indicator words from prompts)
INDICATOR_WORDS_HI = {
    "बेटी", "बेटे", "बेटा", "बच्चे", "बच्चा", "बच्ची",
    "पिता", "माँ", "माता",
    "हिंदू", "मुसलमान", "सिख", "ईसाई", "बौद्ध", "जैन",
    "पारसी", "यहूदी", "नास्तिक", "आदिवासी",
}

INDICATOR_WORDS_EN = {
    "daughter", "son", "child", "children", "girl", "boy",
    "father", "mother", "dad", "mom", "papa", "mama",
    "hindu", "muslim", "sikh", "christian", "buddhist", "jain",
    "parsi", "jewish", "atheist", "tribal",
    "i", "am", "a", "an", "the", "my", "is", "was", "were", "be", "been", "being",
    "and", "or", "but", "so", "to", "of", "in", "on", "at", "for", "with", "by",
    "he", "she", "it", "they", "his", "her", "its", "their", "him", "them",
    "this", "that", "these", "those",
}

INDICATOR_WORDS = INDICATOR_WORDS_HI if LANGUAGE == "hindi" else INDICATOR_WORDS_EN


def tokenize(text: str) -> list[str]:
    """Tokenize text (works for Hindi and English) and drop indicator words."""
    text = re.sub(r'[।\.,!?\"\'\(\)\[\]\{\}—\-:;]', ' ', text)
    if LANGUAGE == "english":
        text = text.lower()
    tokens = text.split()
    return [t.strip() for t in tokens if t.strip() and t not in INDICATOR_WORDS]


def compute_word_bias(stories: list[dict], factor: str, top_k: int = 30) -> dict:
    """
    For each value of `factor`, compute words most positively correlated
    with that value using Pearson correlation.

    Returns: { factor_value: [(word, correlation, count), ...] }
    """
    # Build vocabulary
    vocab_counter = Counter()
    for s in stories:
        if not s.get(factor):
            continue
        tokens = set(tokenize(s["story"]))
        vocab_counter.update(tokens)

    # Keep words appearing in >= 5 stories
    vocab = [w for w, c in vocab_counter.items() if c >= 5]
    word_idx = {w: i for i, w in enumerate(vocab)}

    # Build presence matrix (n_stories × n_words) and labels
    relevant = [s for s in stories if s.get(factor)]
    n = len(relevant)
    factor_values = sorted(set(s[factor] for s in relevant))

    X = np.zeros((n, len(vocab)), dtype=np.float32)
    for i, s in enumerate(relevant):
        for tok in set(tokenize(s["story"])):
            if tok in word_idx:
                X[i, word_idx[tok]] = 1.0

    results = {}
    for value in factor_values:
        # Binary label: is this story from this factor value?
        y = np.array([1.0 if s[factor] == value else 0.0 for s in relevant])
        if y.sum() < 3 or y.sum() == n:
            continue

        # Pearson correlation per word
        y_centered = y - y.mean()
        y_std = y.std()
        word_correlations = []
        for w, j in word_idx.items():
            x = X[:, j]
            x_centered = x - x.mean()
            x_std = x.std()
            if x_std == 0 or y_std == 0:
                continue
            corr = (x_centered * y_centered).mean() / (x_std * y_std)
            count = int(x.sum())
            word_correlations.append((w, float(corr), count))

        # Sort by correlation desc
        word_correlations.sort(key=lambda x: -x[1])
        results[value] = word_correlations[:top_k]

    return results


def print_results(results: dict, factor: str):
    print(f"\n=== Top correlated words by {factor} (Pearson) ===")
    for value, words in results.items():
        top = ", ".join(f"{w}({c:.2f}, n={n})" for w, c, n in words[:8])
        print(f"\n  [{value}]\n    {top}")


def save_results(results: dict, factor: str):
    rows = []
    for value, words in results.items():
        for rank, (word, corr, count) in enumerate(words, 1):
            rows.append({
                "factor": factor,
                "factor_value": value,
                "rank": rank,
                "word": word,
                "pearson_correlation": corr,
                "word_count": count,
            })
    df = pd.DataFrame(rows)
    suffix = "" if LANGUAGE == "hindi" else f"_{LANGUAGE}"
    out_file = OUT_DIR / f"word_bias_{factor}{suffix}.csv"
    df.to_csv(out_file, index=False)
    print(f"  Saved -> {out_file}")


def main():
    print(f"Language: {LANGUAGE}")
    print(f"Loading stories from {STORIES}")
    stories = [json.loads(l) for l in open(STORIES, encoding="utf-8") if l.strip()]
    print(f"  {len(stories)} stories\n")

    for factor in ["child_gender", "religion", "ethnicity", "nationality"]:
        if not any(s.get(factor) for s in stories):
            continue
        print(f"\nProcessing factor: {factor}")
        results = compute_word_bias(stories, factor, top_k=20)
        print_results(results, factor)
        save_results(results, factor)


if __name__ == "__main__":
    main()
