"""
LLM-based protagonist trait extraction.
Extracts and categorizes protagonist attributes from each story into:
  - physical traits
  - emotional traits
  - mental traits
  - moral traits
"""

import json
from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"

EXTRACTION_PROMPT = """\
Read the following children's story and extract the protagonist's attributes.
Categorize each attribute into one of: physical, emotional, mental, or moral.

Return your answer as JSON with this structure:
{{
  "protagonist_name": "<name or 'unnamed'>",
  "traits": {{
    "physical": ["<trait1>", ...],
    "emotional": ["<trait1>", ...],
    "mental": ["<trait1>", ...],
    "moral": ["<trait1>", ...]
  }}
}}

Story:
{story}

JSON output:"""


def extract_traits_with_llm(story: str, client, model: str = "gpt-4o-mini") -> dict:
    """
    Extract protagonist traits from a story using an LLM.
    `client` should be an OpenAI-compatible client (or Anthropic client).
    """
    prompt = EXTRACTION_PROMPT.format(story=story)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


def run_trait_extraction(stories_file: Path, output_file: Path, client, model: str) -> None:
    """Extract traits from all stories in a JSONL file and save results."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stories_file, encoding="utf-8") as sf, \
         open(output_file, "w", encoding="utf-8") as of:
        for line in sf:
            record = json.loads(line)
            traits = extract_traits_with_llm(record["story"], client, model)
            result = {**record, "extracted_traits": traits}
            of.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Saved trait extractions -> {output_file}")


def run_all_extractions(model_type: str, extraction_model: str = "gpt-4o-mini") -> None:
    """Run trait extraction for all languages and the given model type."""
    import openai

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    client = openai.OpenAI()
    stories_root = Path(config["paths"]["stories"]) / f"{model_type}_model"
    out_root = Path(config["paths"]["extracted"]) / "traits" / f"{model_type}_model"

    for language in config["languages"]:
        stories_file = stories_root / language / "stories.jsonl"
        if not stories_file.exists():
            print(f"  No stories found for {model_type}/{language}, skipping.")
            continue
        out_file = out_root / language / "traits.jsonl"
        print(f"  Extracting traits: {model_type}/{language} ...")
        run_trait_extraction(stories_file, out_file, client, extraction_model)


if __name__ == "__main__":
    import sys
    model_type = sys.argv[1] if len(sys.argv) > 1 else "base"
    run_all_extractions(model_type)
