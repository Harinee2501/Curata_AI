# classifier/train.py
# -------------------------------------------------------------------
# Trains a DistilBERT text classifier on the provided dataset.
# Returns the trained model so the bandit can query it for uncertainty
# and verifier confidence scores.
# -------------------------------------------------------------------

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from tqdm import tqdm
import config


def get_tokenizer():
    """Returns the shared tokenizer (call once, reuse everywhere)."""
    return DistilBertTokenizer.from_pretrained(config.CLASSIFIER_MODEL)


def build_model():
    """Instantiates a fresh DistilBERT classifier head."""
    model = DistilBertForSequenceClassification.from_pretrained(
        config.CLASSIFIER_MODEL,
        num_labels=config.NUM_CLASSES,
    )
    return model


def train(model, train_dataset, device="cpu"):
    """
    Fine-tunes the model for config.TRAIN_EPOCHS epochs.

    Parameters
    ----------
    model         : pre-built DistilBERT model (from build_model())
    train_dataset : TextClassificationDataset
    device        : "cuda" if available, else "cpu"

    Returns
    -------
    model  : trained model (in-place update, but also returned for clarity)
    """
    model.to(device)
    model.train()

    loader    = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(config.TRAIN_EPOCHS):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"  epoch {epoch+1}/{config.TRAIN_EPOCHS}", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss    = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"  [train] epoch {epoch+1} — avg loss: {avg_loss:.4f}")

    return model
