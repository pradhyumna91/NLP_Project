"""
Story appropriateness analysis (Section 4 of Biased Tales paper).
Measures text complexity for child-appropriateness.

The paper uses:
- AoA (Average Age of Acquisition) — Kuperman et al. 2012, English-specific
- FKRE (Flesch-Kincaid Reading Ease) — designed for English

For Hindi we adapt these:
- Use word-length / sentence-length proxies as complexity measures
- Average word length (longer = more complex)
- Average sentence length (longer = more complex)
- Lexical density (unique words / total words)
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import pandas as pd

LANGUAGE = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("hindi", "english") else "hindi"

ROOT = Path(__file__).parents[2]
STORIES = ROOT / "data" / "stories" / LANGUAGE / "stories.jsonl"
OUT_DIR = ROOT / "data" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# For English, use real FKRE from textstat
if LANGUAGE == "english":
    import textstat  # type: ignore


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (। = Hindi full stop; . ! ? for English)."""
    sentences = re.split(r'[।\.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize(text: str) -> list[str]:
    text = re.sub(r'[।\.,!?\"\'\(\)\[\]\{\}—\-:;]', ' ', text)
    return [t for t in text.split() if t.strip()]


def compute_complexity(story: str) -> dict:
    """Compute complexity metrics for a single story."""
    sentences = split_sentences(story)
    tokens = tokenize(story)

    if not sentences or not tokens:
        base = {"word_count": 0, "sentence_count": 0,
                "avg_word_length": 0, "avg_sent_length": 0,
                "lexical_density": 0, "complexity_score": 0}
        if LANGUAGE == "english":
            base.update({"flesch_reading_ease": 0, "flesch_kincaid_grade": 0})
        return base

    word_count = len(tokens)
    sentence_count = len(sentences)
    avg_word_length = sum(len(t) for t in tokens) / word_count
    avg_sent_length = word_count / sentence_count
    lexical_density = len(set(tokens)) / word_count

    # Simple composite complexity score (lower = simpler, more child-appropriate)
    complexity_score = (avg_word_length * 1.5) + (avg_sent_length * 0.5)

    result = {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": round(avg_word_length, 2),
        "avg_sent_length": round(avg_sent_length, 2),
        "lexical_density": round(lexical_density, 3),
        "complexity_score": round(complexity_score, 2),
    }

    if LANGUAGE == "english":
        try:
            fkre = textstat.flesch_reading_ease(story)
            fkg = textstat.flesch_kincaid_grade(story)
            result["flesch_reading_ease"] = round(fkre, 2)
            result["flesch_kincaid_grade"] = round(fkg, 2)
        except Exception:
            result["flesch_reading_ease"] = 0
            result["flesch_kincaid_grade"] = 0

    return result


def main():
    print(f"Loading stories from {STORIES}")
    stories = [json.loads(l) for l in open(STORIES, encoding="utf-8") if l.strip()]
    print(f"  {len(stories)} stories\n")

    # Per-story metrics
    rows = []
    for s in stories:
        metrics = compute_complexity(s["story"])
        rows.append({
            "id": s["id"],
            "story_index": s["story_index"],
            "child_gender": s.get("child_gender"),
            "parent_role": s.get("parent_role"),
            "religion": s.get("religion"),
            "ethnicity": s.get("ethnicity"),
            "nationality": s.get("nationality"),
            "template_type": s.get("template_type"),
            **metrics,
        })

    df = pd.DataFrame(rows)
    suffix = "" if LANGUAGE == "hindi" else f"_{LANGUAGE}"
    out_file = OUT_DIR / f"text_complexity_per_story{suffix}.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved per-story metrics -> {out_file}\n")

    # Overall stats
    print("=" * 60)
    print("OVERALL TEXT COMPLEXITY")
    print("=" * 60)
    cols = ["word_count", "sentence_count", "avg_word_length",
            "avg_sent_length", "lexical_density", "complexity_score"]
    if LANGUAGE == "english":
        cols.extend(["flesch_reading_ease", "flesch_kincaid_grade"])
    for col in cols:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) > 1:
            print(f"  {col:22s}: mean={vals.mean():6.2f}  std={vals.std():5.2f}  "
                  f"min={vals.min():5.2f}  max={vals.max():6.2f}")

    # Stats by demographic
    print("\n" + "=" * 60)
    print("COMPLEXITY BY CHILD GENDER")
    print("=" * 60)
    grouped = df.groupby("child_gender").agg({
        "word_count": ["mean", "std"],
        "complexity_score": ["mean", "std"],
        "avg_word_length": "mean",
        "lexical_density": "mean",
    }).round(2)
    print(grouped)

    print("\n" + "=" * 60)
    print("COMPLEXITY BY RELIGION")
    print("=" * 60)
    rel_df = df[df["religion"].notna()]
    if len(rel_df) > 0:
        grouped = rel_df.groupby("religion").agg({
            "word_count": "mean",
            "complexity_score": "mean",
            "avg_word_length": "mean",
        }).round(2).sort_values("complexity_score")
        print(grouped)

    print("\n" + "=" * 60)
    print("APPROPRIATENESS ASSESSMENT")
    print("=" * 60)
    target_words = (220, 320)  # From your generation prompt
    in_range = df[(df["word_count"] >= target_words[0]) & (df["word_count"] <= target_words[1])]
    print(f"  Stories within target word range ({target_words[0]}-{target_words[1]}): "
          f"{len(in_range)}/{len(df)} ({100*len(in_range)/len(df):.1f}%)")

    too_short = df[df["word_count"] < target_words[0]]
    too_long = df[df["word_count"] > target_words[1]]
    print(f"  Too short: {len(too_short)} ({100*len(too_short)/len(df):.1f}%)")
    print(f"  Too long:  {len(too_long)} ({100*len(too_long)/len(df):.1f}%)")


if __name__ == "__main__":
    main()
