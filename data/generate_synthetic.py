# data/generate_synthetic.py
# ─────────────────────────────────────────────────────────────────────
# Generates synthetic rows using Groq-hosted LLMs via OpenAI-compatible API.
#
# Key improvements over v1:
#   1. Async + semaphore: concurrent LLM calls (10x faster)
#   2. Numerical bypass: Iris-style datasets skip the LLM entirely
#      and use fast mixed-strategy sampling instead
#   3. Token truncation: long text fields (e.g. IMDB reviews) are
#      trimmed in few-shot examples to avoid hitting rate limits
#
# Numerical generation change (v2):
#   Old: pure Gaussian N(μ, σ×0.5) — every row looks identical to real
#        data, bandit has zero quality variance to learn from.
#   New: mixed strategy (tight / standard / loose Gaussian + boundary
#        interpolation) so the pool contains a realistic spread of row
#        quality that gives the Thompson bandit a meaningful signal.
# ─────────────────────────────────────────────────────────────────────

import asyncio
import json
import random
import time

import numpy as np
import pandas as pd
from loguru import logger
from openai import AsyncOpenAI, OpenAI

from config import cfg

# ── tunables ──────────────────────────────────────────────────────────
MAX_CONCURRENT_REQUESTS = 10  # semaphore cap; raise carefully on paid tiers
MAX_EXAMPLE_FIELD_CHARS = 300  # truncate long text fields in few-shot context

# ── Numerical strategy mix ────────────────────────────────────────────
# Controls what fraction of the synthetic pool comes from each strategy.
# Adjust to tune the bandit's accept/reject ratio:
#   - More BOUNDARY → more clearly-bad rows → bandit rejects more
#   - More TIGHT    → more clearly-good rows → bandit accepts more
# Target acceptance rate with defaults: ~45–65%
_STRATEGY_WEIGHTS = {
    "tight": 0.30,  # σ × 0.5  — near-real, high verifier confidence
    "standard": 0.35,  # σ × 1.0  — medium quality
    "loose": 0.20,  # σ × 2.0  — exploratory, lower confidence
    "boundary": 0.15,  # interpolated between two different classes — ambiguous
}
# ─────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════
# Dataset-type detection
# ══════════════════════════════════════════════════════════════════════


def _is_numerical_dataset(df: pd.DataFrame, content_cols: list[str]) -> bool:
    """Returns True when every content column is numeric (e.g. Iris, Wine)."""
    return all(pd.api.types.is_numeric_dtype(df[col]) for col in content_cols)


def _avg_text_length(df: pd.DataFrame, content_cols: list[str]) -> float:
    """Mean character length across all text content columns."""
    return df[content_cols].astype(str).apply(lambda c: c.str.len().mean()).mean()


# ══════════════════════════════════════════════════════════════════════
# Fast path: mixed-strategy synthetic generation for numerical datasets
# ══════════════════════════════════════════════════════════════════════


def _generate_numerical_synthetic(
    df: pd.DataFrame,
    label_col: str,
    label_names: list[str],
    content_cols: list[str],
    pool_size: int,
) -> list[dict]:
    """
    Generates synthetic rows for numerical datasets using a mixed strategy
    so the Thompson bandit has meaningful quality variance to learn from.

    Why the old pure-Gaussian approach broke bandit learning
    ────────────────────────────────────────────────────────
    Pure Gaussian sampling (N(μ, σ×0.5) fitted on real data) produces rows
    that are statistically identical to the training set BY DESIGN.  As a
    result:
      • novelty scores cluster near 0.15 — all rows look the same
      • verifier confidence has very little variance across the pool
      • The bandit cannot learn to accept vs reject because there is nothing
        to discriminate between

    Four-strategy pool
    ──────────────────
    tight    (σ×0.5, 30%) — stays close to real data → verifier confident
                             → bandit should learn to ACCEPT
    standard (σ×1.0, 35%) — medium spread, medium confidence
    loose    (σ×2.0, 20%) — drifts from real distribution → lower confidence
                             → bandit starts learning to be selective
    boundary (interp, 15%) — linear interpolation between two rows from
                              DIFFERENT classes, α ∈ [0.3, 0.7].
                              Intentionally ambiguous / mislabeled region.
                              → verifier should say "bad" → bandit learns
                              to REJECT

    Class-conditional sampling
    ──────────────────────────
    Each strategy samples per-class using that class's own μ/σ so synthetic
    class balance mirrors the real dataset.  The old code used a single
    global Gaussian which smeared class boundaries.
    """
    logger.info("Numerical dataset detected — using mixed-strategy sampling (no LLM).")

    # ── Per-class statistics ──────────────────────────────────────────
    class_stats: dict[str, dict] = {}
    for cls in label_names:
        mask = df[label_col].astype(str) == str(cls)
        subset = df.loc[mask, content_cols]
        if subset.empty:
            logger.warning(f"No rows found for label '{cls}' — skipping in stats.")
            continue
        class_stats[cls] = {
            "mean": subset.mean().values.astype(np.float64),
            "std": subset.std(ddof=1)
            .fillna(0)
            .values.clip(min=1e-6)
            .astype(np.float64),
            "rows": subset.values.astype(np.float64),
        }

    available_classes = [c for c in label_names if c in class_stats]
    if not available_classes:
        logger.warning(
            "No class statistics available — falling back to global Gaussian."
        )
        return _fallback_gaussian(df, label_col, label_names, content_cols, pool_size)

    # ── Row counts per strategy ───────────────────────────────────────
    n_counts = {k: max(1, round(v * pool_size)) for k, v in _STRATEGY_WEIGHTS.items()}
    # Fix rounding drift
    drift = pool_size - sum(n_counts.values())
    n_counts["standard"] += drift

    pool: list[dict] = []

    # ── 1. Tight Gaussian — clearly good rows ─────────────────────────
    pool += _class_gaussian(
        class_stats,
        available_classes,
        label_col,
        content_cols,
        n=n_counts["tight"],
        sigma_scale=0.5,
    )

    # ── 2. Standard Gaussian — medium quality ────────────────────────
    pool += _class_gaussian(
        class_stats,
        available_classes,
        label_col,
        content_cols,
        n=n_counts["standard"],
        sigma_scale=1.0,
    )

    # ── 3. Loose Gaussian — exploratory / lower quality ──────────────
    pool += _class_gaussian(
        class_stats,
        available_classes,
        label_col,
        content_cols,
        n=n_counts["loose"],
        sigma_scale=2.0,
    )

    # ── 4. Boundary rows — intentionally ambiguous ────────────────────
    pool += _boundary_rows(
        class_stats,
        available_classes,
        label_col,
        content_cols,
        n=n_counts["boundary"],
    )

    random.shuffle(pool)  # mix strategies so bandit sees varied quality
    logger.info(
        f"Mixed sampling complete: {len(pool)} rows generated "
        f"(tight={n_counts['tight']}, standard={n_counts['standard']}, "
        f"loose={n_counts['loose']}, boundary={n_counts['boundary']})"
    )
    return pool


# ─────────────────────────────────────────────────────────────────────
# Strategy helpers
# ─────────────────────────────────────────────────────────────────────


def _proportional_counts(class_stats: dict, classes: list[str], n: int) -> list[int]:
    """Distribute n across classes proportional to their real-data size."""
    sizes = np.array([len(class_stats[c]["rows"]) for c in classes], dtype=float)
    props = sizes / sizes.sum()
    counts = np.floor(props * n).astype(int)
    # Hand out remainder to the largest classes
    remainder = n - int(counts.sum())
    for i in np.argsort(-props)[:remainder]:
        counts[i] += 1
    return counts.tolist()


def _class_gaussian(
    class_stats: dict,
    classes: list[str],
    label_col: str,
    content_cols: list[str],
    n: int,
    sigma_scale: float,
) -> list[dict]:
    """
    Sample n rows using per-class Gaussian with the given sigma_scale.
    Rows are allocated proportionally to real class sizes.
    """
    rows: list[dict] = []
    counts = _proportional_counts(class_stats, classes, n)

    for cls, cnt in zip(classes, counts):
        if cnt == 0:
            continue
        mu = class_stats[cls]["mean"]
        std = class_stats[cls]["std"] * sigma_scale
        samples = np.random.normal(loc=mu, scale=std, size=(cnt, len(content_cols)))
        for sample in samples:
            row = {col: float(v) for col, v in zip(content_cols, sample)}
            row[label_col] = cls
            rows.append(row)

    return rows


def _boundary_rows(
    class_stats: dict,
    classes: list[str],
    label_col: str,
    content_cols: list[str],
    n: int,
) -> list[dict]:
    """
    Interpolate between randomly chosen rows from TWO DIFFERENT classes.

    α ∈ [0.3, 0.7] puts the synthetic point near the decision boundary —
    intentionally ambiguous so the verifier assigns low confidence and the
    bandit learns to reject these rows.

    Label is assigned to whichever class the interpolation is closer to:
        α < 0.5  → class_a label
        α ≥ 0.5  → class_b label
    """
    rows: list[dict] = []

    if len(classes) < 2:
        # Can't make boundary rows with a single class — use standard Gaussian
        logger.debug(
            "Only one class available — substituting standard Gaussian for boundary rows."
        )
        return _class_gaussian(
            class_stats,
            classes,
            label_col,
            content_cols,
            n=n,
            sigma_scale=1.0,
        )

    for _ in range(n):
        cls_a, cls_b = random.sample(classes, 2)
        rows_a = class_stats[cls_a]["rows"]
        rows_b = class_stats[cls_b]["rows"]
        row_a = rows_a[np.random.randint(len(rows_a))]
        row_b = rows_b[np.random.randint(len(rows_b))]

        alpha = np.random.uniform(0.3, 0.7)  # near the boundary
        interp = (1.0 - alpha) * row_a + alpha * row_b
        label = cls_a if alpha < 0.5 else cls_b

        row = {col: float(v) for col, v in zip(content_cols, interp)}
        row[label_col] = label
        rows.append(row)

    return rows


def _fallback_gaussian(
    df: pd.DataFrame,
    label_col: str,
    label_names: list[str],
    content_cols: list[str],
    pool_size: int,
) -> list[dict]:
    """
    Original single-Gaussian fallback used when per-class stats are
    unavailable (e.g. every class has zero rows after filtering).
    """
    content_df = df[content_cols]
    mu = content_df.mean().values
    std = content_df.std(ddof=1).fillna(1.0).values.clip(min=1e-6)
    pool: list[dict] = []
    for _ in range(pool_size):
        vals = np.random.normal(loc=mu, scale=std)
        row = {col: float(v) for col, v in zip(content_cols, vals)}
        row[label_col] = str(random.choice(label_names))
        pool.append(row)
    return pool


# ══════════════════════════════════════════════════════════════════════
# LLM path helpers
# ══════════════════════════════════════════════════════════════════════


def _build_client() -> AsyncOpenAI:
    """Initialises the async Groq-compatible OpenAI client."""
    return AsyncOpenAI(
        api_key=cfg.LLM_API_KEY,
        base_url=cfg.GROQ_BASE_URL,
    )


def _truncate_row(row: dict, max_chars: int = MAX_EXAMPLE_FIELD_CHARS) -> dict:
    """Truncates long string values so few-shot context stays token-efficient."""
    return {
        k: (v[:max_chars] + "..." if isinstance(v, str) and len(v) > max_chars else v)
        for k, v in row.items()
    }


def _pick_n_examples(avg_len: float) -> int:
    """Fewer examples for long-text datasets to reduce token usage."""
    if avg_len > 500:
        return 1
    if avg_len > 200:
        return 2
    return 3


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
    Truncates long fields to keep prompts lean.
    """
    all_cols = content_cols + [label_col]
    label_rows = df[df[label_col].astype(str) == str(label_value)]

    if len(label_rows) >= n_examples:
        sample = label_rows.sample(n=n_examples, random_state=random.randint(0, 9999))
    else:
        sample = df.sample(
            n=min(n_examples, len(df)), random_state=random.randint(0, 9999)
        )

    raw_rows = sample[all_cols].astype(str).to_dict(orient="records")
    return [_truncate_row(r) for r in raw_rows]


def _build_prompt(
    label_col: str,
    label_value: str,
    content_cols: list[str],
    example_rows: list[dict],
) -> str:
    all_cols = content_cols + [label_col]
    examples_str = "\n".join(
        f"  {json.dumps(row, ensure_ascii=False)}" for row in example_rows
    )
    return f"""You are generating synthetic training data for a machine learning dataset.

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


def _parse_llm_response(response_text: str, all_cols: list[str]) -> dict | None:
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        row = json.loads(text)
        if not all(col in row for col in all_cols):
            missing = [c for c in all_cols if c not in row]
            logger.debug(f"LLM response missing columns: {missing}")
            return None
        return row
    except json.JSONDecodeError as e:
        logger.debug(f"JSON parse error: {e} — response: {text[:200]}")
        return None


# ══════════════════════════════════════════════════════════════════════
# Async generation core
# ══════════════════════════════════════════════════════════════════════


async def _generate_one_row_async(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    label_col: str,
    label_value: str,
    content_cols: list[str],
    example_rows: list[dict],
) -> dict | None:
    """
    Calls the LLM asynchronously under a semaphore to cap concurrency.
    Retries up to cfg.LLM_MAX_RETRIES times on failure.
    """
    all_cols = content_cols + [label_col]
    prompt = _build_prompt(label_col, label_value, content_cols, example_rows)

    async with sem:
        for attempt in range(1, cfg.LLM_MAX_RETRIES + 1):
            try:
                response = await client.chat.completions.create(
                    model=cfg.LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=cfg.LLM_TEMPERATURE,
                    max_tokens=cfg.LLM_MAX_TOKENS,
                )
                text = response.choices[0].message.content
                row = _parse_llm_response(text, all_cols)

                if row is not None:
                    return row
                logger.debug(f"Attempt {attempt}: invalid response, retrying...")

            except Exception as e:
                err_str = str(e).lower()
                if "rate limit" in err_str or "429" in err_str:
                    wait = cfg.LLM_RETRY_WAIT * attempt
                    logger.warning(
                        f"Rate limit hit. Waiting {wait}s (attempt {attempt})..."
                    )
                    await asyncio.sleep(wait)
                elif "invalid api key" in err_str or "401" in err_str:
                    raise EnvironmentError(
                        "Invalid Groq API key. Please check your .env file.\n"
                        "Get a free key at: https://console.groq.com"
                    )
                else:
                    logger.warning(f"LLM error on attempt {attempt}: {e}")
                    await asyncio.sleep(cfg.LLM_RETRY_WAIT)

    logger.warning("All retries exhausted for one row — skipping.")
    return None


async def _generate_pool_async(
    df: pd.DataFrame,
    label_col: str,
    label_names: list[str],
    content_cols: list[str],
    pool_size: int,
    n_examples: int,
) -> list[dict]:
    """Fires all LLM row-generation tasks concurrently under a semaphore."""
    client = _build_client()
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Cycle labels for class balance
    label_cycle = [label_names[i % len(label_names)] for i in range(pool_size)]
    random.shuffle(label_cycle)

    tasks = [
        _generate_one_row_async(
            client,
            sem,
            label_col,
            label_value,
            content_cols,
            _get_example_rows(df, label_col, label_value, content_cols, n_examples),
        )
        for label_value in label_cycle
    ]

    logger.info(
        f"Firing {pool_size} async LLM tasks "
        f"(concurrency cap: {MAX_CONCURRENT_REQUESTS})..."
    )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    pool: list[dict] = []
    failed = 0
    for r in results:
        if isinstance(r, dict):
            pool.append(r)
        else:
            failed += 1

    logger.info(
        f"Async generation complete: {len(pool)} succeeded, {failed} failed "
        f"({len(pool) / pool_size * 100:.1f}% success rate)"
    )
    return pool


# ══════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════


def generate_synthetic_pool(
    df: pd.DataFrame,
    label_col: str,
    label_names: list[str],
    content_cols: list[str],
    target_size: int,
    *,
    pool_multiplier: int | None = None,
    **_: object,
) -> list[dict]:
    """
    Generates a pool of synthetic rows.

    - Numerical datasets  → mixed-strategy sampling (no LLM)
    - Text datasets       → async LLM calls with token-efficient prompts

    Parameters
    ----------
    df             : original user DataFrame
    label_col      : name of the label column
    label_names    : list of unique label values
    content_cols   : all columns except the label column
    target_size    : base row count
    pool_multiplier: overrides cfg.SYNTH_POOL_MULTIPLIER when set
    """
    mult = (
        cfg.SYNTH_POOL_MULTIPLIER if pool_multiplier is None else int(pool_multiplier)
    )
    mult = max(1, mult)
    pool_size = target_size * mult

    logger.info(
        f"Generating synthetic pool of {pool_size} rows "
        f"(target: {target_size}, multiplier: {mult}x)"
    )

    # ── Fast path: numerical datasets ─────────────────────────────────
    if _is_numerical_dataset(df, content_cols):
        pool = _generate_numerical_synthetic(
            df, label_col, label_names, content_cols, pool_size
        )

    # ── LLM path: text / mixed datasets ───────────────────────────────
    else:
        avg_len = _avg_text_length(df, content_cols)
        n_examples = _pick_n_examples(avg_len)
        logger.info(
            f"Text dataset (avg field length: {avg_len:.0f} chars). "
            f"Using {n_examples} few-shot example(s) per prompt."
        )

        pool = asyncio.run(
            _generate_pool_async(
                df, label_col, label_names, content_cols, pool_size, n_examples
            )
        )

    # ── Post-checks ───────────────────────────────────────────────────
    if len(pool) < target_size:
        logger.warning(
            f"Generated only {len(pool)} rows but target is {target_size}. "
            "The bandit will curate from what's available."
        )

    random.shuffle(pool)
    return pool
