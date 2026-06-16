"""
BTC/USD 1-minute LightGBM trainer (target_15m).

Loads the engineered feature dataset, performs a strict sequential
time-series split (70/15/15), trains an LGBMClassifier with early stopping
on the validation slice, and reports precision/recall/F1, a confusion matrix,
and a feature-importance PNG written to outputs/.

Design notes
------------
- Sequential split (no shuffling) preserves the temporal dependency of the
  1-minute bars; random splits would leak future information into training.
- Only engineered indicators are used as features; raw OHLC and Volume are
  excluded so the model cannot memorise the absolute price level and so the
  pipeline is robust to the price scaling seen across 2012-2026.
- LightGBM is given `init_score=False` semantics by using `dataset_params` and
  an explicit `Dataset` for the validation slice, which keeps memory flat.
- Evaluation metrics are computed on the test slice only.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Tuple

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


# ---------- Configuration -----------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
FEATURES_PATH: Path = PROJECT_ROOT / "data" / "processed" / "btcusd_features.csv"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
IMPORTANCE_PNG: Path = OUTPUTS_DIR / "lgbm_feature_importance.png"

# Sequential time-series split (no shuffling).
TRAIN_FRAC: float = 0.70
VAL_FRAC: float = 0.15
TEST_FRAC: float = 0.15
assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9

TARGET_COL: str = "target_15m"

# Engineered indicators only -- excludes raw OHLC + Volume to prevent the
# model from leaning on absolute price level (which would not generalize
# across the multi-order-of-magnitude BTC price history in this dataset).
FEATURE_COLS: List[str] = [
    "log_return",
    "SMA_50", "SMA_200", "EMA_9", "EMA_21",
    "RSI_14",
    "MACD", "MACD_signal", "MACD_hist",
    "BB_mid", "BB_upper", "BB_lower",
]

LGB_PARAMS: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
}
NUM_BOOST_ROUND: int = 1500
EARLY_STOPPING: int = 50
LOG_EVAL: int = 100


# ---------- Logging -----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("train_lgbm")


# ---------- IO + prep ---------------------------------------------------------

def load_dataset(path: Path) -> pd.DataFrame:
    """Read the engineered CSV with a DatetimeIndex; memory-aware dtypes."""
    log.info("Loading features from %s", path)
    df = pd.read_csv(
        path,
        index_col="Timestamp",
        parse_dates=["Timestamp"],
        dtype={c: "float32" for c in FEATURE_COLS},
    )
    # Defensive: drop 'Timestamp' if it shows up as a regular column.
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])
    log.info(
        "Features loaded: shape=%s, memory=%.2f MB",
        df.shape, df.memory_usage(deep=True).sum() / 1e6,
    )
    return df


def time_series_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Strict sequential 70/15/15 split. Returns X_train/val/test, y_*."""
    n = len(df)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    # Train: [0, n_train), Val: [n_train, n_train+n_val), Test: remainder.
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL].astype("int8")
    X_val, y_val = val[FEATURE_COLS], val[TARGET_COL].astype("int8")
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL].astype("int8")

    print(f"[1/5] Split sizes -> "
          f"train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    print(f"      Date ranges -> "
          f"train[{train.index.min()} .. {train.index.max()}]  "
          f"val[{val.index.min()} .. {val.index.max()}]  "
          f"test[{test.index.min()} .. {test.index.max()}]")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------- Model ------------------------------------------------------------

def build_and_train(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
) -> lgb.Booster:
    """Train LightGBM with early stopping on the validation slice."""
    log.info("Training LightGBM with early stopping (patience=%d)...", EARLY_STOPPING)
    t0 = time.time()
    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, free_raw_data=False)
    booster = lgb.train(
        params=LGB_PARAMS,
        train_set=train_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(LOG_EVAL),
        ],
    )
    log.info("Training done in %.1fs. Best iter=%d, best score=%.6f",
             time.time() - t0, booster.best_iteration, booster.best_score["val"]["binary_logloss"])
    return booster


# ---------- Evaluation + plots -----------------------------------------------

def evaluate(booster: lgb.Booster, X_test: pd.DataFrame, y_test: pd.Series) -> np.ndarray:
    """Print classification report + confusion matrix; return predicted labels."""
    print(f"[3/5] Test period rows={len(X_test):,}, "
          f"class balance={dict(y_test.value_counts().sort_index().items())}")
    y_proba = booster.predict(X_test, num_iteration=booster.best_iteration)
    y_pred = (y_proba > 0.5).astype("int8")

    print("\n=== Classification Report (Test) ===")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print("=== Confusion Matrix (Test) ===")
    print(pd.DataFrame(
        cm,
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    ))
    print(f"      TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    return y_pred


def plot_feature_importance(booster: lgb.Booster, importance_type: str = "gain") -> None:
    """Save a horizontal bar chart of the top features to outputs/."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = booster.feature_importance(importance_type=importance_type)
    names = booster.feature_name()
    order = np.argsort(raw)[::-1]
    top = order[:20]  # top-20 keeps the plot readable
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(top))[::-1], raw[top], color="#1f77b4")
    ax.set_yticks(range(len(top))[::-1])
    ax.set_yticklabels([names[i] for i in top])
    ax.set_xlabel(f"Feature importance ({importance_type})")
    ax.set_title("LightGBM Feature Importance - BTC/USD 15m Direction")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(IMPORTANCE_PNG, dpi=140)
    plt.close(fig)
    print(f"[4/5] Feature importance plot -> {IMPORTANCE_PNG}")


# ---------- Orchestration -----------------------------------------------------

def run() -> None:
    df = load_dataset(FEATURES_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = time_series_split(df)

    booster = build_and_train(X_train, y_train, X_val, y_val)
    print(f"[2/5] Best iteration: {booster.best_iteration}")

    _ = evaluate(booster, X_test, y_test)
    plot_feature_importance(booster, importance_type="gain")

    # Persist the trained model for downstream use.
    model_path = OUTPUTS_DIR / "lgbm_model.txt"
    booster.save_model(str(model_path))
    print(f"[5/5] Saved model -> {model_path}")


if __name__ == "__main__":
    run()
