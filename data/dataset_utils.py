# data/dataset_utils.py
# -------------------------------------------------------------------
# Utilities to load real labeled data from HuggingFace and split it
# into the train / validation / test sets used throughout the project.
# -------------------------------------------------------------------

import random
from datasets import load_dataset
from torch.utils.data import Dataset
import config


class TextClassificationDataset(Dataset):
    """
    A simple PyTorch Dataset that wraps a list of (text, label) pairs.
    Used for the real data splits and the curated synthetic pool.
    """

    def __init__(self, texts, labels, tokenizer):
        self.texts  = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=config.MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          self.labels[idx],
        }


def load_real_data():
    """
    Downloads the IMDB dataset (or any dataset set in config) and
    carves out fixed-size train / val / test splits to simulate a
    low-resource scenario.

    Returns
    -------
    train_texts, train_labels  : list[str], list[int]  (500 samples)
    val_texts,   val_labels    : list[str], list[int]  (200 samples)
    test_texts,  test_labels   : list[str], list[int]  (200 samples)
    """
    print(f"[data] Loading '{config.DATASET_NAME}' from HuggingFace ...")
    raw = load_dataset(config.DATASET_NAME)

    # Flatten the original splits into one shuffled pool so we control sizes.
    all_texts  = list(raw["train"]["text"])  + list(raw["test"]["text"])
    all_labels = list(raw["train"]["label"]) + list(raw["test"]["label"])

    combined = list(zip(all_texts, all_labels))
    random.seed(config.RANDOM_SEED)
    random.shuffle(combined)

    total_needed = config.REAL_TRAIN_SIZE + config.VAL_SIZE + config.TEST_SIZE
    combined = combined[:total_needed]

    texts, labels = zip(*combined)

    # Slice into splits
    t = config.REAL_TRAIN_SIZE
    v = t + config.VAL_SIZE

    train_texts,  train_labels  = list(texts[:t]),  list(labels[:t])
    val_texts,    val_labels    = list(texts[t:v]),  list(labels[t:v])
    test_texts,   test_labels   = list(texts[v:]),   list(labels[v:])

    print(f"[data] Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
