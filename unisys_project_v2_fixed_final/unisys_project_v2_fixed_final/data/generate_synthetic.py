# data/generate_synthetic.py
# ─────────────────────────────────────────────────────────────────────
# Generates synthetic rows using Groq-hosted LLMs via OpenAI-compatible API.
#
# For each synthetic sample, the LLM generates values for ALL columns
# (not just the text column), preserving the full structure of the
# user's original dataset.
#
# Output rows have the exact same columns as the input CSV.
# ─────────────────────────────────────────────────────────────────────

import json
import time
import random
import pandas as pd
from openai import OpenAI
from loguru import logger
from config import cfg


def _build_client() -> OpenAI:
    """Initialises the Groq-compatible OpenAI client."""
    return OpenAI(
        api_key=cfg.LLM_API_KEY,
        base_url=cfg.GROQ_BASE_URL,
    )


def _build_prompt(
    label_col: str,
    label_value: str,
    content_cols: list[str],
    example_rows: list[dict],
) -> str:
    """
    Builds a prompt that shows the LLM a few real examples and asks it
    to generate one new row with the same structure.

    Parameters
    ----------
    label_col    : name of the label column
    label_value  : the specific label to generate for (e.g. "positive")
    content_cols : all columns except the label column
    example_rows : a few real rows from the dataset as context
    """
    all_cols = content_cols + [label_col]

    examples_str = "\n".join(
        f"  {json.dumps(row, ensure_ascii=False)}" for row in example_rows
    )

    prompt = f"""You are generating synthetic training data for a machine learning dataset.

Dataset columns: {json.dumps(all_cols)}
Label column: "{label_col}"
Target label for this row: "{label_value}"

Here are some real examples from the dataset:
{examples_str}

Generate exactly ONE new realistic row where "{label_col}" is "{label_value}".
The row must match the style, format, and distribution of the examples above.
All column values must be realistic and consistent with each other.

Respond with ONLY a valid JSON object containing all columns: {json.dumps(all_cols)}
No explanation, no markdown, no extra text — just the raw JSON object."""

    return prompt


def _parse_llm_response(
    response_text: str,
    all_cols: list[str],
) -> dict | None:
    """
    Parses the LLM's JSON response. Returns None if parsing fails.
    """
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text  = "\n".join(lines[1:-1]) if len(lines) > 2 else text

    try:
        row = json.loads(text)
        # Validate all expected columns are present
        if not all(col in row for col in all_cols):
            missing = [c for c in all_cols if c not in row]
            logger.debug(f"LLM response missing columns: {missing}")
            return None
        return row
    except json.JSONDecodeError as e:
        logger.debug(f"JSON parse error: {e} — response was: {text[:200]}")
        return None


def _generate_one_row(
    client: OpenAI,
    label_col: str,
    label_value: str,
    content_cols: list[str],
    example_rows: list[dict],
) -> dict | None:
    """
    Calls the LLM to generate one synthetic row.
    Retries up to cfg.LLM_MAX_RETRIES times on failure.
    """
    all_cols = content_cols + [label_col]
    prompt   = _build_prompt(label_col, label_value, content_cols, example_rows)

    for attempt in range(1, cfg.LLM_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=cfg.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.LLM_TEMPERATURE,
                max_tokens=cfg.LLM_MAX_TOKENS,
            )
            text = response.choices[0].message.content
            row  = _parse_llm_response(text, all_cols)

            if row is not None:
                return row
            else:
                logger.debug(f"Attempt {attempt}: invalid response, retrying...")

        except Exception as e:
            err_str = str(e).lower()
            if "rate limit" in err_str or "429" in err_str:
                wait = cfg.LLM_RETRY_WAIT * attempt
                logger.warning(f"Rate limit hit. Waiting {wait}s before retry {attempt}...")
                time.sleep(wait)
            elif "invalid api key" in err_str or "401" in err_str:
                raise EnvironmentError(
                    "Invalid Groq API key. Please check your .env file.\n"
                    "Get a free key at: https://console.groq.com"
                )
            else:
                logger.warning(f"LLM error on attempt {attempt}: {e}")
                time.sleep(cfg.LLM_RETRY_WAIT)

    logger.warning("All retries exhausted for one row — skipping.")
    return None


def _get_example_rows(
    df: pd.DataFrame,
    label_col: str,
    label_value: str,
    content_cols: list[str],
    n_examples: int = 3,
) -> list[dict]:
    """
    Picks a few real rows with the target label as few-shot examples.
    Falls back to any rows if not enough labeled examples exist.
    """
    all_cols    = content_cols + [label_col]
    label_rows  = df[df[label_col].astype(str) == label_value]

    if len(label_rows) >= n_examples:
        sample = label_rows.sample(n=n_examples, random_state=random.randint(0, 9999))
    else:
        sample = df.sample(n=min(n_examples, len(df)), random_state=random.randint(0, 9999))

    return sample[all_cols].astype(str).to_dict(orient="records")


def generate_synthetic_pool(
    df: pd.DataFrame,
    label_col: str,
    label_names: list[str],
    content_cols: list[str],
    target_size: int,
) -> list[dict]:
    """
    Generates a pool of synthetic rows using the Groq LLM.
    Pool size = target_size * cfg.SYNTH_POOL_MULTIPLIER so the bandit
    has enough candidates to be selective.

    Parameters
    ----------
    df           : original user DataFrame (used for few-shot examples)
    label_col    : name of the label column
    label_names  : list of unique label values (e.g. ["negative", "positive"])
    content_cols : all columns except the label column
    target_size  : how many curated rows the user wants at the end

    Returns
    -------
    pool : list of dicts — each dict has the same keys as the user's CSV columns
    """
    pool_size = target_size * cfg.SYNTH_POOL_MULTIPLIER
    logger.info(
        f"Generating synthetic pool of {pool_size} rows "
        f"(target: {target_size}, multiplier: {cfg.SYNTH_POOL_MULTIPLIER}x)"
    )

    client = _build_client()
    pool   = []
    failed = 0

    # Cycle through labels to maintain class balance in the pool
    label_cycle = [label_names[i % len(label_names)] for i in range(pool_size)]
    random.shuffle(label_cycle)

    for i, label_value in enumerate(label_cycle):
        if i % 50 == 0 and i > 0:
            logger.info(f"  Generated {i}/{pool_size} synthetic rows ({failed} failed)...")

        example_rows = _get_example_rows(df, label_col, label_value, content_cols)
        row = _generate_one_row(client, label_col, label_value, content_cols, example_rows)

        if row is not None:
            pool.append(row)
        else:
            failed += 1

    success = len(pool)
    logger.info(
        f"Synthetic pool complete: {success} generated, {failed} failed "
        f"({success/pool_size*100:.1f}% success rate)"
    )

    if success < target_size:
        logger.warning(
            f"Generated only {success} rows but target is {target_size}. "
            "The bandit will curate from what's available."
        )

    random.shuffle(pool)
    return pool
