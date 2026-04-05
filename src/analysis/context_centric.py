"""
Context-centric analysis.
Examines narrative settings (geographic, urban/rural, socioeconomic) across
demographic groups, languages, and training stages.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"

CONTEXT_CATEGORIES = ["geographic", "urban_rural", "socioeconomic"]


def load_contexts(contexts_file: Path) -> list[dict]:
    records = []
    with open(contexts_file, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def compute_context_distribution(
    records: list[dict],
    group_by: str,
    context_category: str,
) -> dict[str, Counter]:
    """
    Compute context indicator frequency distributions grouped by a demographic attribute.
    """
    distributions: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        group_val = record.get(group_by, "unknown")
        context = record.get("extracted_context", {}).get("context", {})
        for indicator in context.get(context_category, []):
            distributions[group_val][indicator.lower()] += 1
    return dict(distributions)


def run_context_analysis(model_type: str) -> None:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    contexts_root = Path(config["paths"]["extracted"]) / "contexts" / f"{model_type}_model"

    for language in config["languages"]:
        contexts_file = contexts_root / language / "contexts.jsonl"
        if not contexts_file.exists():
            print(f"  No contexts found for {model_type}/{language}, skipping.")
            continue

        records = load_contexts(contexts_file)
        print(f"\n=== {model_type.upper()} | {language} ===")

        for attr in ["religion", "child_gender"]:
            for category in CONTEXT_CATEGORIES:
                dist = compute_context_distribution(records, group_by=attr, context_category=category)
                print(f"\n  [{attr}] {category} context:")
                for group, counter in sorted(dist.items()):
                    top = counter.most_common(5)
                    print(f"    {group}: {top}")


if __name__ == "__main__":
    for model_type in ["base", "rlhf"]:
        run_context_analysis(model_type)
