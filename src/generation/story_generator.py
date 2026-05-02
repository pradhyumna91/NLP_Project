"""
Story generation pipeline using OpenAI API (gpt-4o-mini).
Reads Hindi prompts and generates stories.
Saves results to data/stories/hindi/stories.jsonl
Supports resuming from where it left off.
"""

import json
import os
import time
from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"
ROOT = Path(__file__).parents[2]

# Load .env if present so OPENAI_API_KEY is available
_ENV_PATH = ROOT / ".env"
if _ENV_PATH.exists():
    for _line in _ENV_PATH.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            k, v = _line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

SYSTEM_PROMPTS = {
    "hindi": """\
You are an assistant who writes stories for young children. Write the story entirely in Hindi only. Do not use English or Telugu words, emojis, or special symbols. Do not provide a title. Provide only the story itself. The language should be easy for young children to understand. The story must have a clear beginning, middle, and end. The story should contain only one central event. The emotional connection within the story should feel natural. The story must not end abruptly in the middle. The story should have a gentle and satisfying conclusion. The story length should be approximately between 220 and 320 words.""",
    "english": """\
You are an assistant who writes stories for young children. Write the story entirely in English only. Do not use emojis or special symbols. Do not provide a title. Provide only the story itself. The language should be easy for young children to understand. The story must have a clear beginning, middle, and end. The story should contain only one central event. The emotional connection within the story should feel natural. The story must not end abruptly in the middle. The story should have a gentle and satisfying conclusion. The story length should be approximately between 220 and 320 words.""",
}

PROMPT_FIELD = {"hindi": "prompt_hi", "english": "prompt_en"}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def create_client():
    import openai
    return openai.OpenAI()


def generate_story(prompt: str, client, config: dict, language: str) -> str:
    """Generate a single story using OpenAI API."""
    gen_cfg = config["generation"]
    system_prompt = SYSTEM_PROMPTS[language]

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=config["model"]["name"],
                messages=[
                    {"role": "system", "content": system_prompt},
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


def run_generation(language: str = None) -> None:
    """Generate stories for all prompts in the configured language."""
    config = load_config()
    language = language or config.get("language", "hindi")
    if language not in SYSTEM_PROMPTS:
        raise ValueError(f"Unsupported language: {language}")

    model_name = config["model"]["name"]
    print(f"Using model: {model_name}")
    print(f"Language: {language}")
    client = create_client()

    # Prompts file is always stored under hindi/ (single multilingual file w/ prompt_en + prompt_hi)
    prompts_file = ROOT / config["paths"]["prompts"] / "hindi" / "prompts.jsonl"
    if not prompts_file.exists():
        print(f"Prompts not found at {prompts_file}")
        return

    out_dir = ROOT / config["paths"]["stories"] / language
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "stories.jsonl"

    n_per_config = config["generation"]["stories_per_config"]
    prompt_field = PROMPT_FIELD[language]

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
            prompt = item[prompt_field]
            for i in range(n_per_config):
                if (item["id"], i) in done:
                    continue

                if len(done) % 25 == 0:
                    print(f"  [{len(done)}/{total_stories}] Prompt {item['id']}, story {i}", flush=True)
                story = generate_story(prompt, client, config, language)
                record = {
                    **item,
                    "model_name": model_name,
                    "language": language,
                    "story_index": i,
                    "story": story,
                }
                sf.write(json.dumps(record, ensure_ascii=False) + "\n")
                sf.flush()
                done.add((item["id"], i))

    print(f"Done! {len(done)} stories saved -> {out_file}")


if __name__ == "__main__":
    import sys
    lang = sys.argv[1] if len(sys.argv) > 1 else None
    run_generation(language=lang)
