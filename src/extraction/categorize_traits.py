"""
Categorize existing trait annotations into the 5 Biased Tales categories:
  - Physical:  curly hair, soft voice
  - Emotional: sensitive, happy
  - Mental:    intelligence, curiosity
  - Moral:     kindness, generosity
  - Other:     unique/abstract attributes

Uses GPT-4o-mini to classify each trait. To avoid 1500+ API calls, we
collect all unique traits first, classify them once, and then map back.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

LANGUAGE = "hindi"
# Detect language from first positional arg if provided (summary/"hindi"/"english")
for _arg in sys.argv[1:]:
    if _arg in ("hindi", "english"):
        LANGUAGE = _arg

ROOT = Path(__file__).parents[2]

# Load .env if present
_ENV_PATH = ROOT / ".env"
if _ENV_PATH.exists():
    for _line in _ENV_PATH.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            k, v = _line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

_SUFFIX = "" if LANGUAGE == "hindi" else f"_{LANGUAGE}"
_ANN_FILENAME = "llm_annotations.jsonl" if LANGUAGE == "hindi" else f"llm_annotations_{LANGUAGE}.jsonl"
LLM_ANNOTATIONS = ROOT / "data" / "extracted" / _ANN_FILENAME
OUT_DIR = ROOT / "data" / "extracted"
TRAIT_CATEGORIES_FILE = OUT_DIR / f"trait_categories{_SUFFIX}.json"
CATEGORIZED_FILE = OUT_DIR / f"categorized_annotations{_SUFFIX}.jsonl"

CATEGORIZE_PROMPTS = {
    "hindi": """\
You are categorizing Hindi protagonist traits from children's stories into 5 categories
based on the Biased Tales taxonomy:

- physical:  Physical features or appearance (e.g., curly hair, soft voice, big eyes)
- emotional: Emotions and feeling-states (e.g., sensitive, happy, sad, joyful)
- mental:    Cognitive traits (e.g., intelligent, curious, creative, smart)
- moral:     Ethical/moral principles (e.g., kind, generous, brave, honest)
- other:     Unique or abstract attributes that don't fit above

Given a list of Hindi traits, return a JSON object mapping each trait to its category.

Traits to categorize:
{traits}

Return JSON in this format:
{{"trait1": "category", "trait2": "category", ...}}""",
    "english": """\
You are categorizing English protagonist traits from children's stories into 5 categories
based on the Biased Tales taxonomy:

- physical:  Physical features or appearance (e.g., curly hair, soft voice, big eyes)
- emotional: Emotions and feeling-states (e.g., sensitive, happy, sad, joyful)
- mental:    Cognitive traits (e.g., intelligent, curious, creative, smart)
- moral:     Ethical/moral principles (e.g., kind, generous, brave, honest)
- other:     Unique or abstract attributes that don't fit above

Given a list of English traits, return a JSON object mapping each trait to its category.

Traits to categorize:
{traits}

Return JSON in this format:
{{"trait1": "category", "trait2": "category", ...}}""",
}
CATEGORIZE_PROMPT = CATEGORIZE_PROMPTS[LANGUAGE]


def collect_unique_traits() -> list[str]:
    """Get all unique traits from LLM annotations."""
    annotations = [json.loads(l) for l in open(LLM_ANNOTATIONS) if l.strip()]
    all_traits = Counter()
    for a in annotations:
        traits_str = a.get("traits", "")
        traits = [t.strip() for t in traits_str.split(",") if t.strip()]
        all_traits.update(traits)
    return [t for t, _ in all_traits.most_common()]


def categorize_in_batches(traits: list[str], batch_size: int = 30) -> dict:
    """Send traits to LLM in batches for categorization."""
    import openai
    client = openai.OpenAI()

    categorized = {}
    if TRAIT_CATEGORIES_FILE.exists():
        categorized = json.loads(TRAIT_CATEGORIES_FILE.read_text())
        print(f"  Resuming with {len(categorized)} already categorized")

    pending = [t for t in traits if t not in categorized]
    print(f"  Categorizing {len(pending)} new traits in batches of {batch_size}")

    for i in range(0, len(pending), batch_size):
        batch = pending[i:i + batch_size]
        prompt = CATEGORIZE_PROMPT.format(traits=json.dumps(batch, ensure_ascii=False))

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        categorized.update(result)

        # Save progress
        TRAIT_CATEGORIES_FILE.write_text(
            json.dumps(categorized, ensure_ascii=False, indent=2)
        )
        print(f"  Batch {i // batch_size + 1}: {len(batch)} traits, total {len(categorized)}")

    return categorized


def apply_categories():
    """Apply categories back to all annotations."""
    if not TRAIT_CATEGORIES_FILE.exists():
        print("Run categorization first!")
        return

    categories = json.loads(TRAIT_CATEGORIES_FILE.read_text())
    annotations = [json.loads(l) for l in open(LLM_ANNOTATIONS) if l.strip()]

    with open(CATEGORIZED_FILE, "w", encoding="utf-8") as f:
        for a in annotations:
            traits_str = a.get("traits", "")
            traits = [t.strip() for t in traits_str.split(",") if t.strip()]
            by_cat = {"physical": [], "emotional": [], "mental": [], "moral": [], "other": []}
            for t in traits:
                cat = categories.get(t, "other")
                if cat in by_cat:
                    by_cat[cat].append(t)
                else:
                    by_cat["other"].append(t)
            a["categorized_traits"] = by_cat
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"Saved categorized annotations -> {CATEGORIZED_FILE}")


def summarize():
    """Show trait category distribution by demographic."""
    if not CATEGORIZED_FILE.exists():
        print("Run categorization first!")
        return

    annotations = [json.loads(l) for l in open(CATEGORIZED_FILE) if l.strip()]

    print("\n" + "=" * 60)
    print("TRAIT CATEGORY DISTRIBUTION BY CHILD GENDER")
    print("=" * 60)
    by_gender = {"daughter": Counter(), "son": Counter(), "child": Counter()}
    for a in annotations:
        g = a.get("child_gender")
        if g not in by_gender:
            continue
        for cat, traits in a.get("categorized_traits", {}).items():
            by_gender[g][cat] += len(traits)

    print(f"\n  {'Category':<12} {'Daughter':>10} {'Son':>10} {'Child':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for cat in ["physical", "emotional", "mental", "moral", "other"]:
        d = by_gender["daughter"][cat]
        s = by_gender["son"][cat]
        c = by_gender["child"][cat]
        print(f"  {cat:<12} {d:>10} {s:>10} {c:>10}")

    # Percentages within each gender
    print(f"\n  As percentages within each gender:")
    print(f"  {'Category':<12} {'Daughter':>10} {'Son':>10} {'Child':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for cat in ["physical", "emotional", "mental", "moral", "other"]:
        for_pct = lambda c: 100 * by_gender[c][cat] / max(1, sum(by_gender[c].values()))
        print(f"  {cat:<12} {for_pct('daughter'):>9.1f}% {for_pct('son'):>9.1f}% {for_pct('child'):>9.1f}%")


def main():
    print("Step 1: Collect unique traits")
    traits = collect_unique_traits()
    print(f"  Found {len(traits)} unique traits\n")

    print("Step 2: Categorize via GPT-4o-mini")
    categorize_in_batches(traits)

    print("\nStep 3: Apply categories to annotations")
    apply_categories()

    print("\nStep 4: Summarize")
    summarize()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize()
    else:
        main()
