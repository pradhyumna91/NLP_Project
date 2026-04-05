"""
LLM-based narrative context extraction.
Extracts setting/environment descriptors from each story, categorized as:
  - geographic (e.g., region, country implied)
  - urban_rural (e.g., village, city, forest)
  - socioeconomic (e.g., wealth indicators, occupation)
"""

import json
from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"

EXTRACTION_PROMPT = """\
Read the following children's story and extract the narrative setting details.
Categorize each detail into one of: geographic, urban_rural, or socioeconomic.

Return your answer as JSON with this structure:
{{
  "setting_summary": "<brief description>",
  "context": {{
    "geographic": ["<indicator1>", ...],
    "urban_rural": ["<indicator1>", ...],
    "socioeconomic": ["<indicator1>", ...]
  }}
}}

Story:
{story}

JSON output:"""


def extract_context_with_llm(story: str, client, model: str = "gpt-4o-mini") -> dict:
    """Extract narrative context/setting from a story using an LLM."""
    prompt = EXTRACTION_PROMPT.format(story=story)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


def run_context_extraction(stories_file: Path, output_file: Path, client, model: str) -> None:
    """Extract context from all stories in a JSONL file and save results."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stories_file, encoding="utf-8") as sf, \
         open(output_file, "w", encoding="utf-8") as of:
        for line in sf:
            record = json.loads(line)
            context = extract_context_with_llm(record["story"], client, model)
            result = {**record, "extracted_context": context}
            of.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Saved context extractions -> {output_file}")


def run_all_extractions(model_type: str, extraction_model: str = "gpt-4o-mini") -> None:
    """Run context extraction for all languages and the given model type."""
    import openai

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    client = openai.OpenAI()
    stories_root = Path(config["paths"]["stories"]) / f"{model_type}_model"
    out_root = Path(config["paths"]["extracted"]) / "contexts" / f"{model_type}_model"

    for language in config["languages"]:
        stories_file = stories_root / language / "stories.jsonl"
        if not stories_file.exists():
            print(f"  No stories found for {model_type}/{language}, skipping.")
            continue
        out_file = out_root / language / "contexts.jsonl"
        print(f"  Extracting context: {model_type}/{language} ...")
        run_context_extraction(stories_file, out_file, client, extraction_model)


if __name__ == "__main__":
    import sys
    model_type = sys.argv[1] if len(sys.argv) > 1 else "base"
    run_all_extractions(model_type)
