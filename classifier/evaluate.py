# classifier/evaluate.py
# -------------------------------------------------------------------
# Evaluation utilities:
#   - compute F1-score on a labeled dataset
#   - get predicted probabilities for a list of texts
#     (used by the feature extractors for uncertainty / verifier)
# -------------------------------------------------------------------

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import config


@torch.no_grad()
def evaluate_f1(model, dataset, device="cpu") -> float:
    """
    Runs inference on 'dataset' and returns the macro-averaged F1-score.
    This is the reward signal for the bandit.
    """
    model.eval()
    model.to(device)

    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds  = logits.argmax(dim=-1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(batch["label"].numpy())

    f1 = f1_score(all_labels, all_preds, average="macro")
    return float(f1)


@torch.no_grad()
def get_probabilities(model, tokenizer, texts: list[str], device="cpu") -> np.ndarray:
    """
    Returns a (N, num_classes) array of softmax probabilities for each text.
    Used by:
      - uncertainty.py  → entropy of this distribution
      - verifier.py     → max probability = verifier confidence
    """
    model.eval()
    model.to(device)

    all_probs = []
    batch_size = 32

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoding = tokenizer(
            batch_texts,
            max_length=config.MAX_SEQ_LEN,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

    return np.vstack(all_probs)   # shape: (N, num_classes)
