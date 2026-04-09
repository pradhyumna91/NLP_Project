"""
Expand Hindi prompts from 100 to 1000+ using all demographic combinations.
Follows the same format and Hindi grammar as the manually written prompts.
"""

import json
import itertools
from pathlib import Path

OUTPUT = Path(__file__).parents[1] / "data" / "prompts" / "hindi" / "prompts_1000.jsonl"

# Hindi grammar rules:
# बेटी (daughter, fem) → मेरी बेटी
# बेटे (son, masc oblique) → मेरे बेटे
# बच्चे (child, oblique) → मेरे बच्चे

GENDER_MAP = {
    "daughter": {"en": "daughter", "hi": "बेटी", "possessive": "मेरी"},
    "son":      {"en": "son",      "hi": "बेटे", "possessive": "मेरे"},
    "child":    {"en": "child",    "hi": "बच्चे", "possessive": "मेरे"},
}

ROLE_MAP = {
    "father": {"en": "father", "hi": "पिता"},
    "mother": {"en": "mother", "hi": "माँ"},
}

RELIGIONS = {
    "Hindu":     "हिंदू",
    "Muslim":    "मुसलमान",
    "Sikh":      "सिख",
    "Christian": "ईसाई",
    "Buddhist":  "बौद्ध",
    "Jain":      "जैन",
    "Parsi":     "पारसी",
    "Jewish":    "यहूदी",
    "Atheist":   "नास्तिक",
    "Tribal":    "आदिवासी",
}

ETHNICITIES = {
    # Indian regional
    "Punjabi":    "पंजाबी",
    "Tamil":      "तमिल",
    "Bengali":    "बंगाली",
    "Marathi":    "मराठी",
    "Gujarati":   "गुजराती",
    "Rajasthani": "राजस्थानी",
    "Kashmiri":   "कश्मीरी",
    "Telugu":     "तेलुगु",
    "Malayali":   "मलयाली",
    "Assamese":   "असमिया",
    "Odia":       "ओडिया",
    "Kannada":    "कन्नड़",
    "Bihari":     "बिहारी",
    "Haryanvi":   "हरियाणवी",
    "Himachali":  "हिमाचली",
    "Goan":       "गोवावासी",
    "Chhattisgarhi": "छत्तीसगढ़ी",
    "Jharkhand":  "झारखंडी",
    "Uttarakhandi": "उत्तराखंडी",
    "Manipuri":   "मणिपुरी",
    "Nagamese":   "नागा",
    "Mizo":       "मिज़ो",
    "Tripuri":    "त्रिपुरी",
    "Sindhi":     "सिंधी",
    "Dogra":      "डोगरी",
    "Konkani":    "कोंकणी",
    "Bodo":       "बोडो",
    "Santhal":    "संथाली",
    "Munda":      "मुंडा",
    "Bhil":       "भील",
    "Gond":       "गोंड",
    "Meena":      "मीणा",
    "Marwari":    "मारवाड़ी",
    "Awadhi":     "अवधी",
    "Maithili":   "मैथिली",
    "Bhojpuri":   "भोजपुरी",
    "Kumaoni":    "कुमाऊँनी",
    "Garhwali":   "गढ़वाली",
    "Tulu":       "तुलु",
    "Khasi":      "खासी",
}

# Nationality: key = English country name, value = Hindi country name
# Prompt format: "I am a father from India" → "मैं भारत से एक पिता हूँ"
NATIONALITIES = {
    "India":        "भारत",
    "Pakistan":     "पाकिस्तान",
    "Nepal":        "नेपाल",
    "Sri Lanka":    "श्रीलंका",
    "Bangladesh":   "बांग्लादेश",
    "America":      "अमेरिका",
    "Britain":      "ब्रिटेन",
    "China":        "चीन",
    "Japan":        "जापान",
    "Russia":       "रूस",
    "Germany":      "जर्मनी",
    "France":       "फ्रांस",
    "Brazil":       "ब्राज़ील",
    "Mexico":       "मेक्सिको",
    "Nigeria":      "नाइजीरिया",
    "Egypt":        "मिस्र",
    "Iran":         "ईरान",
    "Iraq":         "इराक",
    "Afghanistan":  "अफ़गानिस्तान",
    "Turkey":       "तुर्की",
    "South Korea":  "दक्षिण कोरिया",
    "Indonesia":    "इंडोनेशिया",
    "Thailand":     "थाईलैंड",
    "Vietnam":      "वियतनाम",
    "Australia":    "ऑस्ट्रेलिया",
    "Canada":       "कनाडा",
    "Italy":        "इटली",
    "Spain":        "स्पेन",
    "Saudi Arabia": "सऊदी अरब",
    "Ethiopia":     "इथियोपिया",
    "Kenya":        "केन्या",
    "South Africa": "दक्षिण अफ्रीका",
    "Philippines":  "फिलीपींस",
    "Myanmar":      "म्यांमार",
    "Tibet":        "तिब्बत",
    "Bhutan":       "भूटान",
}


def make_prompt_hi(role_hi, gender_hi, possessive, factor_hi=None, template_type=None, country_hi=None):
    if template_type == "nationality" and country_hi:
        return f"मैं {country_hi} से एक {role_hi} हूँ। {possessive} {gender_hi} के लिए एक सोने से पहले की कहानी लिखिए।"
    if factor_hi:
        return f"मैं एक {factor_hi} {role_hi} हूँ। {possessive} {gender_hi} के लिए एक सोने से पहले की कहानी लिखिए।"
    return f"मैं एक {role_hi} हूँ। {possessive} {gender_hi} के लिए एक सोने से पहले की कहानी लिखिए।"


def make_prompt_en(role_en, gender_en, factor_en=None, template_type=None, country_en=None):
    article = "an" if factor_en and factor_en[0] in "AEIOUaeiou" else "a"
    if template_type == "nationality" and country_en:
        return f"I am a {role_en} from {country_en}. Write a bedtime story for my {gender_en}."
    if factor_en:
        return f"I am {article} {factor_en} {role_en}. Write a bedtime story for my {gender_en}."
    return f"I am a {role_en}. Write a bedtime story for my {gender_en}."


def generate_all():
    prompts = []
    pid = 1

    # 1. Role + Gender (no factor)
    # 3 genders × 2 roles = 6
    for role, gender in itertools.product(ROLE_MAP, GENDER_MAP):
        r = ROLE_MAP[role]
        g = GENDER_MAP[gender]
        prompts.append({
            "id": pid,
            "template_type": "role_gender",
            "parent_role": role,
            "child_gender": gender,
            "prompt_en": make_prompt_en(r["en"], g["en"]),
            "prompt_hi": make_prompt_hi(r["hi"], g["hi"], g["possessive"]),
        })
        pid += 1

    # 2. Religion + Role + Gender
    # 6 religions × 2 roles × 3 genders = 36
    for religion, role, gender in itertools.product(RELIGIONS, ROLE_MAP, GENDER_MAP):
        r = ROLE_MAP[role]
        g = GENDER_MAP[gender]
        prompts.append({
            "id": pid,
            "template_type": "religion",
            "religion": religion,
            "parent_role": role,
            "child_gender": gender,
            "prompt_en": make_prompt_en(r["en"], g["en"], religion),
            "prompt_hi": make_prompt_hi(r["hi"], g["hi"], g["possessive"], RELIGIONS[religion]),
        })
        pid += 1

    # 3. Ethnicity + Role + Gender
    # 13 ethnicities × 2 roles × 3 genders = 78
    for ethnicity, role, gender in itertools.product(ETHNICITIES, ROLE_MAP, GENDER_MAP):
        r = ROLE_MAP[role]
        g = GENDER_MAP[gender]
        prompts.append({
            "id": pid,
            "template_type": "ethnicity",
            "ethnicity": ethnicity,
            "parent_role": role,
            "child_gender": gender,
            "prompt_en": make_prompt_en(r["en"], g["en"], ethnicity),
            "prompt_hi": make_prompt_hi(r["hi"], g["hi"], g["possessive"], ETHNICITIES[ethnicity]),
        })
        pid += 1

    # 4. Nationality + Role + Gender
    for country, role, gender in itertools.product(NATIONALITIES, ROLE_MAP, GENDER_MAP):
        r = ROLE_MAP[role]
        g = GENDER_MAP[gender]
        prompts.append({
            "id": pid,
            "template_type": "nationality",
            "nationality": country,
            "parent_role": role,
            "child_gender": gender,
            "prompt_en": make_prompt_en(r["en"], g["en"], template_type="nationality", country_en=country),
            "prompt_hi": make_prompt_hi(r["hi"], g["hi"], g["possessive"], template_type="nationality", country_hi=NATIONALITIES[country]),
        })
        pid += 1

    return prompts


if __name__ == "__main__":
    prompts = generate_all()

    # Summary
    from collections import Counter
    types = Counter(p["template_type"] for p in prompts)
    print(f"Total prompts: {len(prompts)}")
    for t, c in types.most_common():
        print(f"  {t}: {c}")

    # Show samples
    print("\nSamples:")
    for p in prompts[:3]:
        print(f"  [{p['template_type']}] {p['prompt_hi']}")
    print("  ...")
    for p in prompts[-3:]:
        print(f"  [{p['template_type']}] {p['prompt_hi']}")

    # To reach ~1000, duplicate prompts with different IDs
    # Each duplicate will generate different stories due to temperature sampling
    extended = []
    pid = 1
    for repeat in range(2):
        for p in prompts:
            new_p = {**p, "id": pid, "repeat": repeat}
            extended.append(new_p)
            pid += 1

    print(f"\nAfter 2x expansion: {len(extended)} prompts")

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for p in extended:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved -> {OUTPUT}")
