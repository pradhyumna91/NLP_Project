"""
Character-centric analysis.
Compares protagonist trait distributions across:
  - demographic conditions (religion, gender, nationality)
  - languages (English, Hindi, Telugu)
  - training stages (base vs RLHF)
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"

TRAIT_CATEGORIES = ["physical", "emotional", "mental", "moral"]


def load_traits(traits_file: Path) -> list[dict]:
    records = []
    with open(traits_file, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def compute_trait_distribution(
    records: list[dict],
    group_by: str,
    trait_category: str,
) -> dict[str, Counter]:
    """
    Compute trait frequency distributions grouped by a demographic attribute.

    group_by: one of 'religion', 'child_gender', 'parent_role'
    trait_category: one of 'physical', 'emotional', 'mental', 'moral'
    """
    distributions: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        group_val = record.get(group_by, "unknown")
        traits = record.get("extracted_traits", {}).get("traits", {})
        for trait in traits.get(trait_category, []):
            distributions[group_val][trait.lower()] += 1
    return dict(distributions)


def top_traits(counter: Counter, n: int = 10) -> list[tuple[str, int]]:
    return counter.most_common(n)


def run_character_analysis(model_type: str) -> None:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    traits_root = Path(config["paths"]["extracted"]) / "traits" / f"{model_type}_model"

    for language in config["languages"]:
        traits_file = traits_root / language / "traits.jsonl"
        if not traits_file.exists():
            print(f"  No traits found for {model_type}/{language}, skipping.")
            continue

        records = load_traits(traits_file)
        print(f"\n=== {model_type.upper()} | {language} ===")

        for attr in ["religion", "child_gender"]:
            for category in TRAIT_CATEGORIES:
                dist = compute_trait_distribution(records, group_by=attr, trait_category=category)
                print(f"\n  [{attr}] {category} traits:")
                for group, counter in sorted(dist.items()):
                    top = top_traits(counter, n=5)
                    print(f"    {group}: {top}")


if __name__ == "__main__":
    for model_type in ["base", "rlhf"]:
        run_character_analysis(model_type)
