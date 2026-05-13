# classifier/train.py
# ─────────────────────────────────────────────────────────────────────
# Trains a Logistic Regression classifier for tabular data
# ─────────────────────────────────────────────────────────────────────

from sklearn.linear_model import LogisticRegression
from loguru import logger
from config import cfg


def get_tokenizer():
    """Not needed for tabular data"""
    return None


def build_model(num_classes: int):
    """Creates a Logistic Regression model"""
    model = LogisticRegression(
        max_iter=3000,
        multi_class="auto",
        n_jobs=-1,
    )
    return model


def train(model, train_dataset, device="cpu"):
    """
    train_dataset = (X, y)
    """
    X, y = train_dataset

    logger.info("Training Logistic Regression model...")
    model.fit(X, y)

    return model