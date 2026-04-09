# RL-Guided Synthetic Data Selection for Data-Centric NLP

**Team:** Amoha V (23011102011) · M Harinee (23011102044)  
**Mentor:** Dr. Sundharakumar K B

---

## Overview

This project implements a contextual bandit framework that intelligently selects
high-quality synthetic training samples for NLP classification.  Three quality
signals drive the agent's decisions:

| Signal | Meaning | Computation |
|--------|---------|-------------|
| Classifier Uncertainty | How unsure the model is about this sample | Entropy of softmax probabilities |
| Semantic Novelty | How different this sample is from real data | Cosine distance to nearest real embedding |
| Verifier Confidence | How likely the synthetic label is correct | Model's probability for the assigned label |

The bandit receives these as a context vector and decides **accept** or **reject**
for each candidate.  The reward is the **ΔF1** on the validation set after retraining.

---

## Project Structure

```
rl_synth_selection/
├── config.py                  ← all hyperparameters (edit here)
├── pipeline.py                ← main training loop
├── plot_results.py            ← visualise results
├── requirements.txt
├── data/
│   ├── dataset_utils.py       ← load real data, PyTorch Dataset class
│   └── generate_synthetic.py ← synthetic pool generation
├── features/
│   ├── uncertainty.py         ← signal 1: entropy
│   ├── novelty.py             ← signal 2: semantic distance
│   └── verifier.py            ← signal 3: label confidence
├── bandit/
│   ├── base_bandit.py         ← abstract base class
│   ├── ucb.py                 ← UCB1 algorithm
│   └── thompson.py            ← Thompson Sampling algorithm
└── classifier/
    ├── train.py               ← DistilBERT fine-tuning
    └── evaluate.py            ← F1 computation, probability extraction
```

---

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python pipeline.py

# 4. Plot the results (after pipeline finishes)
python plot_results.py
```

---

## Switching Between Bandit Algorithms

Open `config.py` and change:

```python
BANDIT_ALGO = "ucb"       # Upper Confidence Bound
# or
BANDIT_ALGO = "thompson"  # Thompson Sampling
```

---

## Plugging in a Real LLM

Open `data/generate_synthetic.py` and replace the `_generate_with_llm()` function:

```python
import openai
client = openai.OpenAI(api_key="YOUR_KEY")

def _generate_with_llm(label: int) -> str:
    sentiment = "positive" if label == 1 else "negative"
    prompt = f"Write a short {sentiment} movie review in 2-3 sentences."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
```

Everything else in the pipeline works unchanged.

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `REAL_TRAIN_SIZE` | 500 | Simulates low-resource scenario |
| `SYNTH_POOL_SIZE` | 1000 | Total LLM-generated candidates |
| `SYNTH_TARGET_SIZE` | 300 | How many to keep after curation |
| `SYNTH_BATCH_SIZE` | 50 | Candidates evaluated per iteration |
| `UCB_ALPHA` | 1.0 | UCB exploration constant |
| `TRAIN_EPOCHS` | 2 | Classifier epochs per iteration |
