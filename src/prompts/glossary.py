"""
Fixed glossary for demographic labels across English and Hindi.
Used to ensure consistent translation of key terms in prompt templates.
Back-translation verification should be run against these terms.
"""

GLOSSARY = {
    # Child gender terms
    "daughter": {
        "english": "daughter",
        "hindi": "बेटी",
    },
    "son": {
        "english": "son",
        "hindi": "बेटा",
    },
    "child": {
        "english": "child",
        "hindi": "बच्चे",
    },
    # Parent role terms
    "father": {
        "english": "father",
        "hindi": "पिता",
    },
    "mother": {
        "english": "mother",
        "hindi": "माँ",
    },
    "parent": {
        "english": "parent",
        "hindi": "अभिभावक",
    },
    # Religion terms
    "Atheist": {
        "english": "Atheist",
        "hindi": "नास्तिक",
    },
    "Muslim": {
        "english": "Muslim",
        "hindi": "मुसलमान",
    },
    "Hindu": {
        "english": "Hindu",
        "hindi": "हिंदू",
    },
    "Christian": {
        "english": "Christian",
        "hindi": "ईसाई",
    },
    "Jewish": {
        "english": "Jewish",
        "hindi": "यहूदी",
    },
    "Buddhist": {
        "english": "Buddhist",
        "hindi": "बौद्ध",
    },
    # Instruction terms
    "bedtime story": {
        "english": "bedtime story",
        "hindi": "सोने की कहानी",
    },
}


def get_term(term: str, language: str) -> str:
    """Look up a demographic term in the given language."""
    if term not in GLOSSARY:
        raise KeyError(f"Term '{term}' not in glossary.")
    if language not in GLOSSARY[term]:
        raise KeyError(f"Language '{language}' not available for term '{term}'.")
    return GLOSSARY[term][language]
