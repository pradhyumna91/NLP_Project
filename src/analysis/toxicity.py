"""
Toxicity analysis using OpenAI Moderation API.
The Biased Tales paper uses Perspective API; we use OpenAI's
omni-moderation-latest which has better Hindi support.

Returns toxicity scores per story across multiple categories.
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

LANGUAGE = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("hindi", "english") else "hindi"

ROOT = Path(__file__).parents[2]

# Load .env if present
_ENV_PATH = ROOT / ".env"
if _ENV_PATH.exists():
    for _line in _ENV_PATH.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            k, v = _line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

STORIES = ROOT / "data" / "stories" / LANGUAGE / "stories.jsonl"
OUT_DIR = ROOT / "data" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
_SUFFIX = "" if LANGUAGE == "hindi" else f"_{LANGUAGE}"
OUT_FILE = OUT_DIR / f"toxicity_scores{_SUFFIX}.jsonl"


def get_client():
    import openai
    return openai.OpenAI()


def score_story(story_text: str, client) -> dict:
    """Get toxicity scores for a single story."""
    for attempt in range(5):
        try:
            response = client.moderations.create(
                model="omni-moderation-latest",
                input=story_text,
            )
            r = response.results[0]
            return {
                "flagged": r.flagged,
                "category_scores": dict(r.category_scores),
                "categories_flagged": [k for k, v in dict(r.categories).items() if v],
            }
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 10 * (attempt + 1)
                print(f"      Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Failed after 5 retries")


def run_toxicity_scoring():
    print(f"Loading stories from {STORIES}")
    stories = [json.loads(l) for l in open(STORIES, encoding="utf-8") if l.strip()]
    print(f"  {len(stories)} stories\n")

    # Resume from existing
    done = set()
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done.add((rec["id"], rec["story_index"]))
        print(f"Resuming: {len(done)} already scored")

    client = get_client()
    with open(OUT_FILE, "a", encoding="utf-8") as out:
        for i, s in enumerate(stories):
            key = (s["id"], s["story_index"])
            if key in done:
                continue

            scores = score_story(s["story"], client)
            rec = {
                "id": s["id"],
                "story_index": s["story_index"],
                "child_gender": s.get("child_gender"),
                "parent_role": s.get("parent_role"),
                "religion": s.get("religion"),
                "ethnicity": s.get("ethnicity"),
                "nationality": s.get("nationality"),
                **scores,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            done.add(key)

            if (i + 1) % 50 == 0:
                print(f"  [{len(done)}/{len(stories)}] processed")

    print(f"\nDone. {len(done)} stories scored -> {OUT_FILE}")


def summarize():
    """Compute summary stats from toxicity scores."""
    if not OUT_FILE.exists():
        print(f"No scores found at {OUT_FILE}. Run toxicity scoring first.")
        return

    records = [json.loads(l) for l in open(OUT_FILE) if l.strip()]
    if not records:
        return

    print("\n" + "=" * 60)
    print(f"TOXICITY SUMMARY ({len(records)} stories)")
    print("=" * 60)

    # Overall flag rate
    flagged = sum(1 for r in records if r.get("flagged"))
    print(f"  Stories flagged: {flagged}/{len(records)} ({100*flagged/len(records):.2f}%)")

    # Average scores per category
    categories = list(records[0].get("category_scores", {}).keys())
    rows = []
    for r in records:
        row = {
            "id": r["id"],
            "story_index": r["story_index"],
            "child_gender": r.get("child_gender"),
            "religion": r.get("religion"),
            "flagged": r.get("flagged"),
        }
        for c in categories:
            row[c] = r.get("category_scores", {}).get(c, 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    tox_csv = OUT_DIR / f"toxicity_per_story{_SUFFIX}.csv"
    df.to_csv(tox_csv, index=False)
    print(f"  Saved per-story scores -> {tox_csv}")

    print("\n  Average toxicity by category:")
    for c in categories:
        avg = df[c].mean()
        max_val = df[c].max()
        print(f"    {c:30s} avg={avg:.5f}  max={max_val:.4f}")

    # By gender
    if "child_gender" in df.columns and df["child_gender"].notna().any():
        print("\n  Average overall toxicity by child gender:")
        # Use any harassment/violence-like category for comparison
        rel_cats = [c for c in categories if "violence" in c or "harassment" in c]
        if rel_cats:
            grouped = df.groupby("child_gender")[rel_cats].mean().round(5)
            print(grouped)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize()
    else:
        run_toxicity_scoring()
        summarize()
