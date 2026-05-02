# RL-Guided Synthetic Data Selection

Intelligently augments your NLP dataset using a Thompson Sampling bandit
that learns which LLM-generated rows are actually worth keeping.

---

## How It Works

1. You provide a labeled CSV dataset
2. The system generates synthetic candidate rows via Groq LLM
3. A Thompson Sampling bandit scores each row using 3 quality signals:
   - **Classifier Uncertainty** — is this sample near a decision boundary?
   - **Semantic Novelty** — does this sample cover new territory?
   - **Verifier Confidence** — is the label likely correct?
4. Only rows that improve model performance are kept
5. Output is a curated set of synthetic rows in the exact same structure as your CSV

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 1b. (Optional) Run Streamlit frontend
```bash
streamlit run streamlit_app.py
```

### 2. Set up your API key
```bash
cp .env.example .env
# Edit .env and add your Groq API key (free at https://console.groq.com)
```

### 3. Run the pipeline
```bash
# Minimal — will ask which column is the label
python pipeline.py --csv your_data.csv --target 200

# Full — specify label column directly
python pipeline.py --csv your_data.csv --label-col sentiment --target 200
```

---

## Input Format

Any CSV file with at least 80 rows. Example:

```
review,                          rating,  sentiment
"Amazing product, love it",      5,       positive
"Broke after one week",          1,       negative
...
```

- **Label column** — the column you want to classify (you specify this)
- **Everything else** — all other columns are treated as content and will be reproduced in the output

---

## Output

All outputs are saved to the `outputs/` folder:

| File | Description |
|---|---|
| `curated_synthetic_samples.csv` | New synthetic rows only — same columns as your CSV |
| `augmented_dataset.csv` | Your original data + curated synthetic rows combined |
| `experiment_report.json` | Full metrics, reward history, bandit statistics |
| `pipeline.log` | Detailed run log |

---

## Configuration

Edit `config.yaml` to adjust pipeline behaviour:

```yaml
train_epochs: 2           # increase for better classifier quality
synth_pool_multiplier: 3  # pool = target × multiplier
synth_batch_size: 50      # samples evaluated per bandit iteration
early_stop_patience: 5    # stop if F1 doesn't improve for N iterations
```

Edit `.env` to change the LLM:

```
LLM_API_KEY=gsk_xxxxxxxxxxxx
LLM_MODEL=llama-3.1-8b-instant   # or llama-3.3-70b-versatile
```

---

## Supported Groq Models

| Model | Speed | Quality |
|---|---|---|
| `llama-3.1-8b-instant` | Very fast | Good (recommended) |
| `llama-3.3-70b-versatile` | Moderate | Best |
| `mixtral-8x7b-32768` | Fast | Good, long context |

---

## Requirements

- Python 3.10+
- Free Groq API key — https://console.groq.com
- Minimum 80 rows in your CSV
- At least 2 unique label values
