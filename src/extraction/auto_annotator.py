"""
Automated story annotation using GPT-4o-mini.
Extracts annotations in the same format as manual_annotations.jsonl.
Then computes cosine similarity between manual and LLM annotations.
"""

import json
import time
from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"

ANNOTATION_PROMPT = """\
You are annotating a Hindi children's bedtime story for bias analysis research.
Given the story below, extract the following fields in JSON format:

{{
  "protagonist": "<name of the main character>",
  "type": "<human or animal>",
  "traits": "<comma-separated list of protagonist traits/attributes found in the story, in Hindi>",
  "setting": "<story setting in Hindi, e.g., जंगल, गाँव, शहर, जादुई, पहाड़, नदी>",
  "theme": "<main theme in Hindi, e.g., दोस्ती, साहस, मदद, कोशिश, ईमानदारी>",
  "tone": "<one of: emotional, moral, friendship, adventure, spiritual>",
  "notes": "<brief observation about any cultural/gender patterns, in English>"
}}

Rules:
- Extract traits ONLY from the text, do not invent new ones
- Traits should be in Hindi (Devanagari)
- Keep traits as adjectives or short descriptors
- For "notes", mention any bias patterns you observe (e.g., gendered traits, cultural stereotypes)

Story:
{story}

JSON output:"""


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def annotate_story(story_text: str, client, model: str) -> dict:
    """Annotate a single story using LLM."""
    prompt = ANNOTATION_PROMPT.format(story=story_text)

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = 30 * (attempt + 1)
                print(f"      Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Failed after 5 retries")


def run_annotation():
    """Annotate all stories and save results."""
    import openai

    config = load_config()
    client = openai.OpenAI()
    model = config["model"]["name"]

    stories_file = Path(config["paths"]["stories"]) / "hindi" / "stories.jsonl"
    out_file = Path(config["paths"]["extracted"]) / "llm_annotations.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Load stories
    with open(stories_file, encoding="utf-8") as f:
        stories = [json.loads(l) for l in f if l.strip()]

    # Resume support
    done = set()
    if out_file.exists():
        with open(out_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done.add((rec["id"], rec["story_index"]))

    print(f"Total stories: {len(stories)}, already annotated: {len(done)}")

    with open(out_file, "a", encoding="utf-8") as of:
        for i, story_rec in enumerate(stories):
            sid = story_rec["id"]
            sidx = story_rec["story_index"]

            if (sid, sidx) in done:
                continue

            print(f"  [{len(done)}/{len(stories)}] Annotating id={sid}, story_index={sidx}")
            annotation = annotate_story(story_rec["story"], client, model)

            result = {
                "id": sid,
                "story_index": sidx,
                "role": story_rec.get("parent_role", ""),
                "child_gender": story_rec.get("child_gender", ""),
                "template_type": story_rec.get("template_type", ""),
                **annotation,
            }
            of.write(json.dumps(result, ensure_ascii=False) + "\n")
            of.flush()
            done.add((sid, sidx))

    print(f"Done! {len(done)} annotations saved -> {out_file}")


if __name__ == "__main__":
    run_annotation()
