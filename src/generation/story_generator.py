"""
Story generation pipeline using OpenAI API (gpt-4o-mini).
Reads Hindi prompts and generates stories.
Saves results to data/stories/hindi/stories.jsonl
Supports resuming from where it left off.
"""

import json
import time
from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"

SYSTEM_PROMPT = """\
You are a children's story writer. Write a bedtime story in Hindi based on the user's prompt.
The story should be:
- Written entirely in Hindi (Devanagari script)
- Simple, age-appropriate, and non-toxic
- 300-500 words long
- Can feature human children, animals with names, or fictional characters
- Should have a moral or lesson
- End with a goodnight message matching the parent role and child gender from the prompt
Write only the story, no translations or explanations."""


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def create_client():
    import openai
    return openai.OpenAI()


def generate_story(prompt: str, client, config: dict) -> str:
    """Generate a single story using OpenAI API."""
    gen_cfg = config["generation"]

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=config["model"]["name"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=gen_cfg["temperature"],
                top_p=gen_cfg["top_p"],
                max_tokens=gen_cfg["max_new_tokens"],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 30 * (attempt + 1)
                print(f"      Rate limited, waiting {wait}s (attempt {attempt+1}/5)...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Failed after 5 retries")


def run_generation() -> None:
    """Generate stories for all Hindi prompts."""
    config = load_config()
    model_name = config["model"]["name"]
    print(f"Using model: {model_name}")
    client = create_client()

    prompts_file = Path(config["paths"]["prompts"]) / "hindi" / "prompts.jsonl"
    if not prompts_file.exists():
        print(f"Prompts not found at {prompts_file}")
        return

    out_dir = Path(config["paths"]["stories"]) / "hindi"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "stories.jsonl"

    n_per_config = config["generation"]["stories_per_config"]

    # Load existing stories to support resuming
    done = set()
    if out_file.exists():
        with open(out_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done.add((rec["id"], rec["story_index"]))
        if done:
            print(f"Resuming: {len(done)} stories already generated")

    # Load all prompts
    with open(prompts_file, encoding="utf-8") as f:
        all_prompts = [json.loads(l) for l in f if l.strip()]
    total_stories = len(all_prompts) * n_per_config
    print(f"Total: {len(all_prompts)} prompts × {n_per_config} stories = {total_stories}")
    print(f"Remaining: {total_stories - len(done)}")

    with open(out_file, "a", encoding="utf-8") as sf:
        for item in all_prompts:
            prompt = item["prompt_hi"]
            for i in range(n_per_config):
                if (item["id"], i) in done:
                    continue

                print(f"  [{len(done)}/{total_stories}] Prompt {item['id']}, story {i}")
                story = generate_story(prompt, client, config)
                record = {
                    **item,
                    "model_name": model_name,
                    "story_index": i,
                    "story": story,
                }
                sf.write(json.dumps(record, ensure_ascii=False) + "\n")
                sf.flush()
                done.add((item["id"], i))

    print(f"Done! {len(done)} stories saved -> {out_file}")


if __name__ == "__main__":
    run_generation()
