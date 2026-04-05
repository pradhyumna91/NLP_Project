"""
Compare manual vs LLM annotations using cosine similarity.
Uses sentence-transformers (all-MiniLM-L6-v2) following the Biased Tales paper.
"""

import json
from pathlib import Path

import numpy as np


EXTRACTED_DIR = Path(__file__).parents[2] / "data" / "extracted"


def load_annotations(filepath: Path) -> list[dict]:
    """Load annotations from JSONL file."""
    annotations = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))
    return annotations


def get_embeddings(texts: list[str], model):
    """Get sentence embeddings for a list of texts."""
    return model.encode(texts, convert_to_numpy=True)


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def run_comparison():
    from sentence_transformers import SentenceTransformer

    manual_file = EXTRACTED_DIR / "manual_annotations.jsonl"
    llm_file = EXTRACTED_DIR / "llm_annotations.jsonl"

    if not manual_file.exists():
        print(f"Manual annotations not found: {manual_file}")
        return
    if not llm_file.exists():
        print(f"LLM annotations not found: {llm_file}")
        return

    manual = load_annotations(manual_file)
    llm = load_annotations(llm_file)

    # Build lookup for LLM annotations by (id, story_index)
    llm_lookup = {(a["id"], a["story_index"]): a for a in llm}

    # Load multilingual model for Hindi support
    print("Loading sentence-transformers model...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Compare overlapping annotations
    trait_similarities = []
    setting_similarities = []
    theme_similarities = []
    overall_similarities = []

    print(f"\n{'ID':>3} {'SI':>2} {'Gender':>8} | {'Trait Sim':>9} {'Setting Sim':>11} {'Theme Sim':>9} | {'Overall':>7}")
    print("-" * 75)

    for m in manual:
        key = (m["id"], m["story_index"])
        if key not in llm_lookup:
            continue

        l = llm_lookup[key]

        # Compare traits
        m_traits = m.get("traits", "")
        l_traits = l.get("traits", "")
        if m_traits and l_traits:
            emb = get_embeddings([m_traits, l_traits], model)
            sim = cosine_similarity(emb[0], emb[1])
            trait_similarities.append(sim)
        else:
            sim = 0.0
            trait_similarities.append(sim)
        t_sim = sim

        # Compare settings
        m_setting = m.get("setting", "")
        l_setting = l.get("setting", "")
        if m_setting and l_setting:
            emb = get_embeddings([m_setting, l_setting], model)
            sim = cosine_similarity(emb[0], emb[1])
            setting_similarities.append(sim)
        else:
            sim = 0.0
            setting_similarities.append(sim)
        s_sim = sim

        # Compare themes
        m_theme = m.get("theme", "")
        l_theme = l.get("theme", "")
        if m_theme and l_theme:
            emb = get_embeddings([m_theme, l_theme], model)
            sim = cosine_similarity(emb[0], emb[1])
            theme_similarities.append(sim)
        else:
            sim = 0.0
            theme_similarities.append(sim)
        th_sim = sim

        overall = (t_sim + s_sim + th_sim) / 3
        overall_similarities.append(overall)

        print(f"{m['id']:>3} {m['story_index']:>2} {m['child_gender']:>8} | {t_sim:>9.4f} {s_sim:>11.4f} {th_sim:>9.4f} | {overall:>7.4f}")

    if not trait_similarities:
        print("\nNo overlapping annotations found!")
        return

    print("-" * 75)
    print(f"{'AVERAGE':>14} | {np.mean(trait_similarities):>9.4f} {np.mean(setting_similarities):>11.4f} {np.mean(theme_similarities):>9.4f} | {np.mean(overall_similarities):>7.4f}")

    print(f"\n=== Summary ===")
    print(f"Stories compared: {len(trait_similarities)}")
    print(f"Avg trait similarity:   {np.mean(trait_similarities):.4f}")
    print(f"Avg setting similarity: {np.mean(setting_similarities):.4f}")
    print(f"Avg theme similarity:   {np.mean(theme_similarities):.4f}")
    print(f"Avg overall similarity: {np.mean(overall_similarities):.4f}")

    # Save results
    results = {
        "n_compared": len(trait_similarities),
        "avg_trait_similarity": float(np.mean(trait_similarities)),
        "avg_setting_similarity": float(np.mean(setting_similarities)),
        "avg_theme_similarity": float(np.mean(theme_similarities)),
        "avg_overall_similarity": float(np.mean(overall_similarities)),
    }
    out_file = EXTRACTED_DIR / "annotation_comparison.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out_file}")


if __name__ == "__main__":
    run_comparison()
