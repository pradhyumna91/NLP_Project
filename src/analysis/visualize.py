"""
Generate visualizations from the bias analysis results.
Saves charts to data/figures/

Usage:
    python -m src.analysis.visualize hindi      # default
    python -m src.analysis.visualize english
    python -m src.analysis.visualize compare    # cross-lingual side-by-side
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans']

ROOT = Path(__file__).parents[2]
OUT_DIR = ROOT / "data" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR = ROOT / "data" / "analysis"


def paths_for(language: str):
    ann_name = "llm_annotations.jsonl" if language == "hindi" else f"llm_annotations_{language}.jsonl"
    return {
        "annotations": ROOT / "data" / "extracted" / ann_name,
        "stories": ROOT / "data" / "stories" / language / "stories.jsonl",
        "predictability_csv": ANALYSIS_DIR / ("predictability_nn_results.csv" if language == "hindi"
                                              else f"predictability_nn_results_{language}.csv"),
    }


def load_data(language: str):
    p = paths_for(language)
    anns = [json.loads(l) for l in open(p["annotations"]) if l.strip()]
    stories = [json.loads(l) for l in open(p["stories"]) if l.strip()]
    story_lookup = {(s['id'], s['story_index']): s for s in stories}
    merged = []
    for a in anns:
        key = (a.get('id'), a.get('story_index'))
        s = story_lookup.get(key, {})
        merged.append({**a, **{k: v for k, v in s.items() if k not in a}})
    return merged


def suffix(language: str) -> str:
    return "" if language == "hindi" else f"_{language}"


def plot_gender_traits(data, language: str):
    daughter, son = Counter(), Counter()
    for d in data:
        traits = [t.strip() for t in d.get('traits', '').split(',') if t.strip()]
        if d.get('child_gender') == 'daughter':
            daughter.update(traits)
        elif d.get('child_gender') == 'son':
            son.update(traits)

    top = sorted(set(t for t, _ in daughter.most_common(15)) | set(t for t, _ in son.most_common(15)))
    d_counts = [daughter[t] for t in top]
    s_counts = [son[t] for t in top]

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(top))
    w = 0.4
    ax.barh(x - w/2, d_counts, w, label='Daughter', color='#e91e63')
    ax.barh(x + w/2, s_counts, w, label='Son', color='#2196f3')
    ax.set_yticks(x)
    ax.set_yticklabels(top)
    ax.set_xlabel('Count')
    ax.set_title(f'Top Protagonist Traits: Daughter vs Son ({language.capitalize()})')
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    out = OUT_DIR / f"01_gender_traits{suffix(language)}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out.name}")


def plot_setting_distribution(data, language: str):
    settings_by_gender = {'daughter': Counter(), 'son': Counter(), 'child': Counter()}
    for d in data:
        gender = d.get('child_gender')
        if gender in settings_by_gender:
            s = d.get('setting', 'unknown').strip()
            if language == "english":
                s = s.lower()
            settings_by_gender[gender][s] += 1

    top_settings = [s for s, _ in (
        settings_by_gender['daughter'] + settings_by_gender['son']
    ).most_common(8)]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(top_settings))
    w = 0.27
    for i, (gender, color) in enumerate([('daughter', '#e91e63'), ('son', '#2196f3'), ('child', '#9e9e9e')]):
        counts = [settings_by_gender[gender][s] for s in top_settings]
        ax.bar(x + (i - 1) * w, counts, w, label=gender, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(top_settings, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Count')
    ax.set_title(f'Story Setting Distribution by Child Gender ({language.capitalize()})')
    ax.legend()
    plt.tight_layout()
    out = OUT_DIR / f"02_setting_distribution{suffix(language)}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out.name}")


def plot_predictability_results(language: str):
    """Bar chart from real predictability CSV — TF-IDF + MLP (matching paper Table 2)."""
    csv_path = paths_for(language)["predictability_csv"]
    df = pd.read_csv(csv_path)
    df = df[df["target"].isin(["child_gender", "parent_role", "religion", "ethnicity", "nationality"])]
    targets = df["target"].tolist()
    labels = [t.replace("_", " ").title() for t in targets]
    baseline = (df["majority_baseline"] * 100).tolist()
    mlp = (df["mlp_accuracy"] * 100).tolist()

    x = np.arange(len(targets))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w/2, baseline, w, label='Majority baseline', color='#bdbdbd')
    ax.bar(x + w/2, mlp,      w, label='TF-IDF + Feed-forward NN (5-fold CV)', color='#3f51b5')

    for i, (b, m) in enumerate(zip(baseline, mlp)):
        ax.text(i - w/2, b + 1, f'{b:.0f}%', ha='center', fontsize=9)
        ax.text(i + w/2, m + 1, f'{m:.0f}%', ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Demographic Predictability from Story Text ({language.capitalize()})')
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    out = OUT_DIR / f"03_predictability{suffix(language)}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out.name}")


def plot_religion_themes(data, language: str):
    religions = ['Hindu', 'Muslim', 'Sikh', 'Christian', 'Buddhist', 'Jain', 'Parsi', 'Jewish', 'Atheist', 'Tribal']
    themes_by_religion = {r: Counter() for r in religions}

    for d in data:
        religion = d.get('religion')
        if religion in themes_by_religion:
            themes = [t.strip() for t in d.get('theme', '').split(',') if t.strip()]
            if language == "english":
                themes = [t.lower() for t in themes]
            themes_by_religion[religion].update(themes)

    all_themes = Counter()
    for c in themes_by_religion.values():
        all_themes.update(c)
    top_themes = [t for t, _ in all_themes.most_common(6)]

    matrix = np.array([
        [themes_by_religion[r][t] for t in top_themes]
        for r in religions
    ])
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_norm = np.divide(matrix, row_sums, where=row_sums != 0) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix_norm, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(np.arange(len(top_themes)))
    ax.set_xticklabels(top_themes, rotation=35, ha='right')
    ax.set_yticks(np.arange(len(religions)))
    ax.set_yticklabels(religions)
    ax.set_title(f'Theme Distribution Across Religions, % within religion ({language.capitalize()})')

    for i in range(len(religions)):
        for j in range(len(top_themes)):
            ax.text(j, i, f'{matrix_norm[i, j]:.0f}', ha='center', va='center',
                    color='black' if matrix_norm[i, j] < 50 else 'white', fontsize=9)

    plt.colorbar(im, ax=ax, label='Percentage')
    plt.tight_layout()
    out = OUT_DIR / f"04_religion_themes{suffix(language)}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out.name}")


def plot_tone_by_gender(data, language: str):
    """Tone by gender — pie chart per gender (daughter / son / child)."""
    tones_by_gender = {'daughter': Counter(), 'son': Counter(), 'child': Counter()}
    for d in data:
        g = d.get('child_gender')
        if g in tones_by_gender:
            tones_by_gender[g][d.get('tone', 'unknown').strip().lower()] += 1

    # Consistent color per tone across the three pies
    tone_colors = {
        'moral':      '#ff9800',
        'friendship': '#4caf50',
        'emotional':  '#e91e63',
        'adventure':  '#2196f3',
        'spiritual':  '#9c27b0',
        'unknown':    '#bdbdbd',
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, gender in zip(axes, ['daughter', 'son', 'child']):
        items = tones_by_gender[gender].most_common()
        labels = [k for k, _ in items]
        sizes = [v for _, v in items]
        colors = [tone_colors.get(l, '#bdbdbd') for l in labels]
        if sum(sizes) == 0:
            ax.set_title(f'{gender.capitalize()} — no data')
            ax.axis('off')
            continue
        ax.pie(sizes, labels=labels, autopct='%1.0f%%', colors=colors,
               startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax.set_title(f'{gender.capitalize()} stories — tone ({language.capitalize()})')

    plt.tight_layout()
    out = OUT_DIR / f"05_tone_by_gender{suffix(language)}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out.name}")


# Reference values from Biased Tales paper (Rooein et al. 2025), Table 4
PAPER_TRAIT_PCT = {
    "Avg":     {"physical": 12.7, "emotional": 29.3, "mental": 34.2, "moral": 19.0, "other": 4.9},
    "GPT-4o":  {"physical": 12.2, "emotional": 30.4, "mental": 34.5, "moral": 20.0, "other": 2.9},
    "Llama3":  {"physical": 19.1, "emotional": 26.3, "mental": 33.1, "moral": 13.4, "other": 8.2},
    "Mixtral": {"physical": 6.5,  "emotional": 31.3, "mental": 35.0, "moral": 23.9, "other": 3.3},
}


def plot_trait_categories(data, language: str):
    """Trait category distribution — single 100% stacked bar combining all child
    genders (daughter + son + child) for the language."""
    cat_filename = "categorized_annotations.jsonl" if language == "hindi" else f"categorized_annotations_{language}.jsonl"
    cat_path = ROOT / "data" / "extracted" / cat_filename
    if not cat_path.exists():
        print(f"  skip 06_trait_categories: {cat_path} missing")
        return

    overall = Counter()
    for line in open(cat_path):
        a = json.loads(line)
        if a.get("child_gender") not in ("daughter", "son", "child"):
            continue
        for cat, traits in a.get("categorized_traits", {}).items():
            overall[cat] += len(traits)

    cats = ["physical", "emotional", "mental", "moral", "other"]
    cat_colors = {"physical": "#e91e63", "emotional": "#ff9800",
                  "mental": "#3f51b5", "moral": "#4caf50", "other": "#9e9e9e"}
    total = sum(overall.values()) or 1
    pcts = [100 * overall[c] / total for c in cats]

    fig, ax = plt.subplots(figsize=(5, 7))
    bottom = 0.0
    for c, v in zip(cats, pcts):
        ax.bar([f'{language.capitalize()}\n(daughter+son+child)'], [v], bottom=[bottom],
               label=c.capitalize(), color=cat_colors[c], edgecolor='white',
               linewidth=1.2, width=0.5)
        if v >= 2:
            ax.text(0, bottom + v / 2, f'{v:.1f}%', ha='center', va='center',
                    color='white' if c in ("mental", "moral", "physical") else 'black',
                    fontsize=11, fontweight='bold')
        bottom += v

    ax.set_ylim(0, 100)
    ax.set_ylabel('% of trait instances (column sums to 100%)')
    ax.set_title(f'Trait Categories ({language.capitalize()}) — Biased Tales Table 4')
    ax.legend(loc='upper right', bbox_to_anchor=(1.55, 1.0))
    plt.tight_layout()
    out = OUT_DIR / f"06_trait_categories{suffix(language)}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out.name}")


def plot_compare_predictability():
    """Cross-lingual comparison: classifier accuracy English vs Hindi."""
    en_csv = ANALYSIS_DIR / "predictability_nn_results_english.csv"
    hi_csv = ANALYSIS_DIR / "predictability_nn_results.csv"
    if not (en_csv.exists() and hi_csv.exists()):
        print("  skip compare_predictability: need both CSVs")
        return

    en = pd.read_csv(en_csv).set_index("target")
    hi = pd.read_csv(hi_csv).set_index("target")
    targets = ["child_gender", "parent_role", "religion", "ethnicity", "nationality"]
    targets = [t for t in targets if t in en.index and t in hi.index]
    labels = [t.replace("_", " ").title() for t in targets]

    baseline = [en.loc[t, "majority_baseline"] * 100 for t in targets]
    en_mlp = [en.loc[t, "mlp_accuracy"] * 100 for t in targets]
    hi_mlp = [hi.loc[t, "mlp_accuracy"] * 100 for t in targets]

    x = np.arange(len(targets))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w, baseline, w, label='Majority baseline', color='#bdbdbd')
    ax.bar(x,     en_mlp,   w, label='English (MLP)', color='#1976d2')
    ax.bar(x + w, hi_mlp,   w, label='Hindi (MLP)',   color='#e91e63')

    for i, (b, e, h) in enumerate(zip(baseline, en_mlp, hi_mlp)):
        ax.text(i - w, b + 1, f'{b:.0f}%', ha='center', fontsize=8)
        ax.text(i,     e + 1, f'{e:.0f}%', ha='center', fontsize=8, fontweight='bold')
        ax.text(i + w, h + 1, f'{h:.0f}%', ha='center', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Classifier accuracy (%)')
    ax.set_title('Demographic Predictability from Story Text — English vs Hindi')
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    out = OUT_DIR / "07_compare_predictability.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out.name}")


def plot_compare_trait_categories():
    """Cross-lingual trait categories — paper Table 4 style.

    Categories on x-axis; grouped bars per (language × gender).
    Each gender has its English bar next to its Hindi bar so the appearance-bias
    asymmetry is directly visible.
    """
    en_path = ROOT / "data" / "extracted" / "categorized_annotations_english.jsonl"
    hi_path = ROOT / "data" / "extracted" / "categorized_annotations.jsonl"
    if not (en_path.exists() and hi_path.exists()):
        print("  skip compare_trait_categories: need both files")
        return

    cats = ["physical", "emotional", "mental", "moral", "other"]
    cat_colors = {"physical": "#e91e63", "emotional": "#ff9800",
                  "mental": "#3f51b5", "moral": "#4caf50", "other": "#9e9e9e"}

    def aggregate(path):
        overall = Counter()
        for line in open(path):
            a = json.loads(line)
            if a.get("child_gender") not in ("daughter", "son", "child"):
                continue
            for cat, traits in a.get("categorized_traits", {}).items():
                overall[cat] += len(traits)
        total = sum(overall.values()) or 1
        return [100 * overall[c] / total for c in cats]

    en = aggregate(en_path)
    hi = aggregate(hi_path)

    # Two 100% stacked columns: English overall, Hindi overall
    labels = ['English\n(daughter+son+child)', 'Hindi\n(daughter+son+child)']
    cols = [en, hi]

    fig, ax = plt.subplots(figsize=(7.5, 7))
    bottom = np.zeros(2)
    for ci, c in enumerate(cats):
        vals = [col[ci] for col in cols]
        ax.bar(labels, vals, bottom=bottom, label=c.capitalize(),
               color=cat_colors[c], edgecolor='white', linewidth=1.2, width=0.55)
        for i, v in enumerate(vals):
            if v >= 2:
                ax.text(i, bottom[i] + v / 2, f'{v:.1f}%', ha='center', va='center',
                        color='white' if c in ("mental", "moral", "physical") else 'black',
                        fontsize=10, fontweight='bold')
        bottom += np.array(vals)

    ax.set_ylim(0, 100)
    ax.set_ylabel('% of trait instances (column sums to 100%)')
    ax.set_title('Trait Categories — English vs Hindi (100% stacked, Biased Tales Table 4)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.34, 1.0))
    plt.tight_layout()
    out = OUT_DIR / "08_compare_trait_categories.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out.name}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "hindi"

    if mode == "compare":
        print(f"Generating cross-lingual comparison figures in {OUT_DIR}/")
        plot_compare_predictability()
        plot_compare_trait_categories()
    else:
        language = mode
        print(f"Loading {language} data...")
        data = load_data(language)
        print(f"  {len(data)} annotated stories\n")
        print(f"Generating {language} figures in {OUT_DIR}/")
        plot_gender_traits(data, language)
        plot_setting_distribution(data, language)
        plot_predictability_results(language)
        plot_religion_themes(data, language)
        plot_tone_by_gender(data, language)
        plot_trait_categories(data, language)

    print(f"\nDone. {len(list(OUT_DIR.glob('*.png')))} total figures in {OUT_DIR}.")


if __name__ == "__main__":
    main()
