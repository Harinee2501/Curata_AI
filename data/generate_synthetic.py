# data/generate_synthetic.py
# -------------------------------------------------------------------
# Generates the synthetic candidate pool.
#
# In production you would call an LLM API (OpenAI, Anthropic, etc.)
# to generate diverse labeled text samples.  For this project we
# simulate synthetic generation by paraphrasing real samples with
# simple lexical perturbations so the code runs without an API key.
#
# To plug in a real LLM, replace _generate_with_llm() below with
# an actual API call — the rest of the pipeline is unchanged.
# -------------------------------------------------------------------

import json
import random
import os
import config


# ── Simple paraphrase templates (stand-in for LLM generation) ───────

POSITIVE_TEMPLATES = [
    "This was a truly {adv} {adj} experience. {praise}",
    "I {verb} this {noun} — {adj} in every way.",
    "What a {adj} piece of work. {praise}",
    "{praise} Definitely worth your {time_noun}.",
    "An {adj} {noun} that {verb_phrase}.",
]

NEGATIVE_TEMPLATES = [
    "This was a {adv} {adj} experience. {criticism}",
    "I {neg_verb} this {noun} — {adj} and forgettable.",
    "What a {adj} waste of time. {criticism}",
    "{criticism} Not worth anyone's {time_noun}.",
    "A {adj} {noun} that {neg_verb_phrase}.",
]

FILL = {
    "adv":           ["genuinely", "completely", "surprisingly", "absolutely", "rather"],
    "adj_pos":       ["wonderful", "brilliant", "captivating", "outstanding", "heartwarming"],
    "adj_neg":       ["terrible", "dreadful", "boring", "disappointing", "forgettable"],
    "praise":        ["Highly recommended.", "A must-see.", "Left me wanting more.", "Exceptional work."],
    "criticism":     ["Avoid at all costs.", "A total letdown.", "Deeply disappointing.", "Not worth it."],
    "verb":          ["loved", "adored", "enjoyed", "appreciated"],
    "neg_verb":      ["hated", "disliked", "despised", "regretted watching"],
    "noun":          ["film", "story", "experience", "production"],
    "time_noun":     ["time", "attention", "evening"],
    "verb_phrase":   ["stayed with me long after", "exceeded all my expectations", "moved me deeply"],
    "neg_verb_phrase": ["wasted two hours of my life", "failed on every level", "put me to sleep"],
}


def _fill_template(template: str, label: int) -> str:
    """Fills one template string with random vocabulary."""
    adj_key = "adj_pos" if label == 1 else "adj_neg"
    result = template
    for key, options in FILL.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, random.choice(options), 1)
    # Replace generic {adj} using sentiment-appropriate list
    result = result.replace("{adj}", random.choice(FILL[adj_key]))
    return result


def _generate_with_llm(label: int) -> str:
    """
    ── REPLACE THIS FUNCTION WITH A REAL LLM CALL ──────────────────

    Example using OpenAI:
        import openai
        client = openai.OpenAI(api_key="YOUR_KEY")
        prompt = f"Write a short {'positive' if label==1 else 'negative'} movie review."
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    For now we use the template-based generator above.
    """
    templates = POSITIVE_TEMPLATES if label == 1 else NEGATIVE_TEMPLATES
    template = random.choice(templates)
    return _fill_template(template, label)


def generate_synthetic_pool(save_path: str = config.SYNTH_DATA_PATH) -> list[dict]:
    """
    Generates SYNTH_POOL_SIZE synthetic (text, label) pairs and saves
    them as a JSON file so we don't regenerate on every run.

    Returns a list of dicts: [{"text": ..., "label": ...}, ...]
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        print(f"[synth] Found existing pool at '{save_path}', loading ...")
        with open(save_path) as f:
            return json.load(f)

    print(f"[synth] Generating {config.SYNTH_POOL_SIZE} synthetic samples ...")
    random.seed(config.RANDOM_SEED)
    pool = []

    for i in range(config.SYNTH_POOL_SIZE):
        # Balanced classes — alternate labels to avoid accidental imbalance
        label = i % config.NUM_CLASSES
        text  = _generate_with_llm(label)
        pool.append({"text": text, "label": label})

    with open(save_path, "w") as f:
        json.dump(pool, f, indent=2)

    print(f"[synth] Saved pool to '{save_path}'")
    return pool
