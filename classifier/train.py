# classifier/train.py
# ─────────────────────────────────────────────────────────────────────
# Trains a classifier for tabular data.
# Supported models (set via config.yaml → classifier_model):
#   - logistic_regression
#   - naive_bayes
#   - random_forest
#   - xgboost
#   - lightgbm
#   - svm
# ─────────────────────────────────────────────────────────────────────

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from loguru import logger
from config import cfg


def get_tokenizer():
    """Not needed for tabular data."""
    return None


def build_model(num_classes: int):
    """
    Instantiates the classifier specified by cfg.CLASSIFIER_MODEL.

    Supported values
    ----------------
    logistic_regression  : Fast linear baseline; well-calibrated probabilities.
    naive_bayes          : Very fast; works well on high-dimensional sparse data.
    random_forest        : Robust ensemble; handles non-linearity and noisy features.
    xgboost              : Gradient boosting via XGBoost (requires `pip install xgboost`).
    lightgbm             : Gradient boosting via LightGBM (requires `pip install lightgbm`).
    svm                  : Support Vector Machine with RBF kernel; strong on small datasets.
    """
    model_key = cfg.CLASSIFIER_MODEL.lower().strip()
    logger.info(f"Building model: '{model_key}' (num_classes={num_classes})")

    if model_key == "logistic_regression":
        return LogisticRegression(
            max_iter=3000,
            multi_class="auto",
            n_jobs=-1,
        )

    if model_key == "naive_bayes":
        # GaussianNB assumes continuous features; swap for MultinomialNB or
        # BernoulliNB in config if your features are counts or binary.
        return GaussianNB()

    if model_key == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,  # Grow full trees; regularise via min_samples_leaf.
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=cfg.RANDOM_SEED,
        )

    if model_key == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError(
                "[train] XGBoost is not installed. Run: pip install xgboost"
            )
        return XGBClassifier(
            n_estimators=300,
            learning_rate=cfg.LEARNING_RATE * 500,  # XGB lr is on a different scale.
            max_depth=6,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=cfg.RANDOM_SEED,
            n_jobs=-1,
        )

    if model_key == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "[train] LightGBM is not installed. Run: pip install lightgbm"
            )
        return lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=cfg.LEARNING_RATE * 500,  # LGB lr is on a different scale.
            num_leaves=63,
            random_state=cfg.RANDOM_SEED,
            n_jobs=-1,
        )

    if model_key == "svm":
        # probability=True enables predict_proba via Platt scaling (slightly slower).
        return SVC(
            kernel="rbf",
            probability=True,
            random_state=cfg.RANDOM_SEED,
        )

    raise ValueError(
        f"[train] Unknown classifier_model '{cfg.CLASSIFIER_MODEL}'. "
        "Choose one of: logistic_regression, naive_bayes, random_forest, "
        "xgboost, lightgbm, svm."
    )


def train(model, train_dataset, device="cpu"):
    """
    Fits *model* on *train_dataset*.

    Parameters
    ----------
    model         : sklearn-compatible estimator returned by build_model().
    train_dataset : tuple (X, y) — numpy arrays or pandas DataFrame/Series.
    device        : Unused; kept for API compatibility with neural backends.
    """
    X, y = train_dataset
    logger.info(f"Training {type(model).__name__} on {len(y)} samples …")
    model.fit(X, y)
    logger.info("Training complete.")
    return model
