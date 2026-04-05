"""
Back-translation verification for translated prompts.
Translated prompts are independently translated back to English and compared
to the original to verify meaning, demographic labels, role info, and
instructional intent are preserved.
"""

import json
from pathlib import Path


VALIDATION_CHECKLIST = [
    "demographic_labels_preserved",   # religion, nationality present
    "role_information_preserved",     # father/mother preserved
    "instructional_intent_preserved", # "write a bedtime story" intent clear
]


def verify_back_translation(
    original_en: str,
    back_translated_en: str,
    checklist: list[str] | None = None,
) -> dict:
    """
    Structured validation of a back-translated prompt.
    In practice, this can be automated with an LLM or done via human spot-check.
    Returns a dict with pass/fail for each checklist item.
    """
    if checklist is None:
        checklist = VALIDATION_CHECKLIST

    # Placeholder: in practice, use an LLM or string-matching heuristics
    results = {item: None for item in checklist}
    results["original"] = original_en
    results["back_translated"] = back_translated_en
    results["note"] = "Fill in pass/fail manually or via LLM verification."
    return results


def load_and_verify_prompts(prompts_file: Path, back_translations_file: Path) -> list[dict]:
    """
    Load original prompts and their back-translations, run verification, return results.

    prompts_file: JSONL with fields: language, religion, parent_role, child_gender, prompt
    back_translations_file: JSONL with field: back_translated_prompt (aligned by line)
    """
    results = []
    with open(prompts_file, encoding="utf-8") as pf, \
         open(back_translations_file, encoding="utf-8") as bf:
        for p_line, b_line in zip(pf, bf):
            prompt_item = json.loads(p_line)
            back_item = json.loads(b_line)
            verification = verify_back_translation(
                original_en=prompt_item.get("original_english", ""),
                back_translated_en=back_item.get("back_translated_prompt", ""),
            )
            results.append({**prompt_item, "verification": verification})
    return results
