"""
Bias analysis on LLM-annotated Hindi stories.
Three analyses following Biased Tales (Rooein et al., EMNLP 2025):
  1. Character-centric: trait distribution across demographics
  2. Context-centric: setting distribution across demographics
  3. Predictability: TF-IDF + classifier to predict demographic from story text
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).parents[2] / "data"


def load_annotations() -> list[dict]:
    filepath = DATA_DIR / "extracted" / "llm_annotations.jsonl"
    with open(filepath, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_stories() -> list[dict]:
    filepath = DATA_DIR / "stories" / "hindi" / "stories.jsonl"
    with open(filepath, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ──────────────────────────────────────────────
# 1. Character-Centric Analysis
# ──────────────────────────────────────────────

def character_centric_analysis(annotations: list[dict]):
    """Analyze trait distributions across child_gender, religion, ethnicity."""
    print("\n" + "=" * 60)
    print("1. CHARACTER-CENTRIC ANALYSIS")
    print("=" * 60)

    # Group traits by demographic factors
    for factor in ["child_gender", "role", "template_type"]:
        print(f"\n--- Traits by {factor} ---")
        trait_counts = defaultdict(Counter)

        for ann in annotations:
            group = ann.get(factor, "unknown")
            traits = [t.strip() for t in ann.get("traits", "").split(",") if t.strip()]
            for trait in traits:
                trait_counts[group][trait] += 1

        for group, counts in sorted(trait_counts.items()):
            total = sum(counts.values())
            top_traits = counts.most_common(8)
            top_str = ", ".join(f"{t}({c})" for t, c in top_traits)
            print(f"  {group:>15}: [{total} total] {top_str}")

    # Gender-specific trait comparison
    print(f"\n--- Daughter vs Son: Top trait comparison ---")
    daughter_traits = Counter()
    son_traits = Counter()
    for ann in annotations:
        traits = [t.strip() for t in ann.get("traits", "").split(",") if t.strip()]
        if ann.get("child_gender") == "daughter":
            daughter_traits.update(traits)
        elif ann.get("child_gender") == "son":
            son_traits.update(traits)

    # Traits unique or much more common for daughters
    print(f"\n  Daughter-leaning traits (appear more in daughter stories):")
    for trait, count in daughter_traits.most_common(15):
        son_count = son_traits.get(trait, 0)
        if count > son_count * 1.3:
            print(f"    {trait}: daughter={count}, son={son_count}")

    print(f"\n  Son-leaning traits (appear more in son stories):")
    for trait, count in son_traits.most_common(15):
        daughter_count = daughter_traits.get(trait, 0)
        if count > daughter_count * 1.3:
            print(f"    {trait}: son={count}, daughter={daughter_count}")

    # Religion-specific traits
    print(f"\n--- Traits by religion ---")
    religion_traits = defaultdict(Counter)
    for ann in annotations:
        if ann.get("template_type") == "religion":
            # Get religion from stories file mapping
            religion_traits[ann.get("role", "unknown")]  # placeholder

    # Extract religion from original stories
    stories = load_stories()
    story_lookup = {}
    for s in stories:
        story_lookup[(s["id"], s["story_index"])] = s

    religion_trait_counts = defaultdict(Counter)
    ethnicity_trait_counts = defaultdict(Counter)
    for ann in annotations:
        key = (ann["id"], ann["story_index"])
        story = story_lookup.get(key, {})
        religion = story.get("religion", "")
        ethnicity = story.get("ethnicity", "")
        traits = [t.strip() for t in ann.get("traits", "").split(",") if t.strip()]

        if religion:
            religion_trait_counts[religion].update(traits)
        if ethnicity:
            ethnicity_trait_counts[ethnicity].update(traits)

    if religion_trait_counts:
        print(f"\n  By Religion:")
        for rel, counts in sorted(religion_trait_counts.items()):
            top = counts.most_common(6)
            top_str = ", ".join(f"{t}({c})" for t, c in top)
            print(f"    {rel:>12}: {top_str}")

    if ethnicity_trait_counts:
        print(f"\n  By Ethnicity:")
        for eth, counts in sorted(ethnicity_trait_counts.items()):
            top = counts.most_common(6)
            top_str = ", ".join(f"{t}({c})" for t, c in top)
            print(f"    {eth:>12}: {top_str}")


# ──────────────────────────────────────────────
# 2. Context-Centric Analysis
# ──────────────────────────────────────────────

def context_centric_analysis(annotations: list[dict]):
    """Analyze setting/theme distributions across demographics."""
    print("\n" + "=" * 60)
    print("2. CONTEXT-CENTRIC ANALYSIS")
    print("=" * 60)

    for factor in ["child_gender", "role"]:
        print(f"\n--- Settings by {factor} ---")
        setting_counts = defaultdict(Counter)
        theme_counts = defaultdict(Counter)
        tone_counts = defaultdict(Counter)

        for ann in annotations:
            group = ann.get(factor, "unknown")
            setting_counts[group][ann.get("setting", "unknown")] += 1
            theme_counts[group][ann.get("theme", "unknown")] += 1
            tone_counts[group][ann.get("tone", "unknown")] += 1

        for group in sorted(setting_counts.keys()):
            print(f"\n  [{group}]")
            print(f"    Settings: {dict(setting_counts[group].most_common(5))}")
            print(f"    Themes:   {dict(theme_counts[group].most_common(5))}")
            print(f"    Tones:    {dict(tone_counts[group].most_common(5))}")

    # Religion-wise settings
    stories = load_stories()
    story_lookup = {(s["id"], s["story_index"]): s for s in stories}

    print(f"\n--- Settings by Religion ---")
    religion_settings = defaultdict(Counter)
    for ann in annotations:
        key = (ann["id"], ann["story_index"])
        story = story_lookup.get(key, {})
        religion = story.get("religion", "")
        if religion:
            religion_settings[religion][ann.get("setting", "unknown")] += 1

    for rel, counts in sorted(religion_settings.items()):
        print(f"  {rel:>12}: {dict(counts.most_common(5))}")


# ──────────────────────────────────────────────
# 3. Predictability Analysis
# ──────────────────────────────────────────────

def predictability_analysis(stories: list[dict]):
    """TF-IDF + Logistic Regression to predict demographics from story text."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.dummy import DummyClassifier

    print("\n" + "=" * 60)
    print("3. PREDICTABILITY ANALYSIS (TF-IDF + Logistic Regression)")
    print("=" * 60)

    # Remove obvious indicator words
    remove_words = ["बेटी", "बेटे", "बेटा", "बच्चे", "बच्चा", "daughter", "son", "child",
                     "लड़की", "लड़का", "girl", "boy"]

    def clean_text(text):
        for w in remove_words:
            text = text.replace(w, "")
        return text

    targets = {
        "child_gender": [s for s in stories if s.get("child_gender") in ("daughter", "son")],
        "role": [s for s in stories],
    }

    # Add religion if present
    religion_stories = [s for s in stories if s.get("religion")]
    if religion_stories:
        targets["religion"] = religion_stories

    # Add ethnicity if present
    ethnicity_stories = [s for s in stories if s.get("ethnicity")]
    if ethnicity_stories:
        targets["ethnicity"] = ethnicity_stories

    for target_name, target_stories in targets.items():
        texts = [clean_text(s["story"]) for s in target_stories]
        labels = [s.get(target_name, "unknown") for s in target_stories]

        # Check minimum class sizes and at least 2 classes
        label_counts = Counter(labels)
        if len(label_counts) < 2:
            print(f"\n  {target_name}: Skipped (only 1 class)")
            continue
        min_count = min(label_counts.values())
        if min_count < 3:
            print(f"\n  {target_name}: Skipped (min class size {min_count} < 3)")
            continue

        n_splits = min(5, min_count)

        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(texts)
        y = np.array(labels)

        # Majority baseline
        dummy = DummyClassifier(strategy="most_frequent")
        dummy_scores = cross_val_score(dummy, X, y, cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42), scoring="accuracy")

        # LR classifier
        clf = LogisticRegression(max_iter=1000, random_state=42)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

        print(f"\n  Target: {target_name}")
        print(f"    Classes: {dict(label_counts)}")
        print(f"    Majority baseline: {dummy_scores.mean():.1%}")
        print(f"    TF-IDF + LR ({n_splits}-fold CV): {scores.mean():.1%} (±{scores.std():.1%})")
        print(f"    Bias signal: {'STRONG' if scores.mean() > dummy_scores.mean() + 0.1 else 'MODERATE' if scores.mean() > dummy_scores.mean() + 0.05 else 'WEAK'}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("Loading data...")
    annotations = load_annotations()
    stories = load_stories()
    print(f"Loaded {len(annotations)} annotations, {len(stories)} stories")

    character_centric_analysis(annotations)
    context_centric_analysis(annotations)
    predictability_analysis(stories)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
