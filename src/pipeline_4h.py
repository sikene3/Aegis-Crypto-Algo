"""
BTC/USD 4-Hour end-to-end pipeline.

Stages
------
    1. Resample 1m cleaned OHLCV -> 4H bars (O=first, H=max, L=min, C=last, V=sum).
    2. Engineer features: log_return, SMA_50, SMA_200, RSI_14, MACD (line/signal/hist),
       BB_upper, BB_lower.  Target = 1 if Close_{t+1} > Close_t else 0.
    3. Train LightGBM on the first 70% (with 15% val for early stopping),
       evaluate on the held-out 15% test slice.
    4. Backtest V3 on the test slice: long-only when P(up) > 0.55, hold 1 bar (4h),
       0.1% fee per side, fixed-fractional sizing.

All stages are modular functions, called from a single `run()`. No external
TA library is used; indicators are pure pandas/numpy.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import List, Tuple

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


# ---------- Configuration -----------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CLEAN_PATH: Path = PROJECT_ROOT / "data" / "processed" / "btcusd_cleaned.csv"
FEATURES_4H_PATH: Path = PROJECT_ROOT / "data" / "processed" / "btcusd_features_4h.csv"
MODEL_PATH: Path = PROJECT_ROOT / "outputs" / "lgbm_model_4h.txt"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
EQUITY_PNG: Path = OUTPUTS_DIR / "portfolio_growth_4h.png"

# Resampling
RESAMPLE_FREQ: str = "4h"

# Train/val/test split (sequential)
TRAIN_FRAC: float = 0.70
VAL_FRAC: float = 0.15
TEST_FRAC: float = 0.15
assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9

TARGET_COL: str = "target_1"
TARGET_HORIZON: int = 1  # 1 bar = 4 hours at the 4H timeframe

# Engineered feature set (NO raw OHLCV; only indicators the model must rely on)
FEATURE_COLS: List[str] = [
    "log_return",
    "SMA_50", "SMA_200",
    "RSI_14",
    "MACD", "MACD_signal", "MACD_hist",
    "BB_upper", "BB_lower",
]

# LightGBM hyperparameters
LGB_PARAMS: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
}
NUM_BOOST_ROUND: int = 2000
EARLY_STOPPING: int = 100
LOG_EVAL: int = 100

# Backtest
INITIAL_CAPITAL: float = 10_000.0
FEE_PER_SIDE: float = 0.001
FEE_ROUND_TRIP: float = 2.0 * FEE_PER_SIDE
PROB_THRESHOLD: float = 0.55
RISK_FRACTION: float = 1.0   # 4H horizon with single position: full equity
HOURS_PER_YEAR: int = 24 * 365


# ---------- Logging -----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pipeline_4h")


# =============================================================================
# 1) Resample
# =============================================================================

def resample_to_4h(clean_path: Path) -> pd.DataFrame:
    log.info("[1/8] Loading 1m cleaned data from %s", clean_path)
    df = pd.read_csv(
        clean_path,
        index_col="Timestamp",
        parse_dates=["Timestamp"],
        dtype={
            "Open": "float32", "High": "float32", "Low": "float32",
            "Close": "float32", "Volume": "float32",
        },
    )
    log.info("1m shape=%s, memory=%.2f MB",
             df.shape, df.memory_usage(deep=True).sum() / 1e6)

    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    bars = df.resample(RESAMPLE_FREQ).agg(agg).dropna()
    log.info("4H shape=%s  [%s .. %s]",
             bars.shape, bars.index.min(), bars.index.max())
    return bars


# =============================================================================
# 2) Features
# =============================================================================

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def build_features_4h(bars: pd.DataFrame) -> pd.DataFrame:
    log.info("[2/8] Engineering 4H features")
    df = bars.copy()
    close = df["Close"]

    df["log_return"] = np.log(close / close.shift(1))
    df["SMA_50"] = close.rolling(window=50, min_periods=50).mean()
    df["SMA_200"] = close.rolling(window=200, min_periods=200).mean()
    df["RSI_14"] = _rsi(close, 14)

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["MACD"] = macd_line
    df["MACD_signal"] = signal_line
    df["MACD_hist"] = macd_line - signal_line

    mid = close.rolling(window=20, min_periods=20).mean()
    sd = close.rolling(window=20, min_periods=20).std(ddof=0)
    df["BB_upper"] = mid + 2.0 * sd
    df["BB_lower"] = mid - 2.0 * sd

    # Forward target: 1 if next bar's Close > current Close.
    df[TARGET_COL] = (close.shift(-TARGET_HORIZON) > close).astype("int8")

    n0 = len(df)
    df = df.dropna()
    n1 = len(df)
    log.info("Features built: %d -> %d rows (dropped %d warm-up/target NaNs)",
             n0, n1, n0 - n1)
    return df


def save_features(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True, index_label="Timestamp", float_format="%.6f")
    log.info("Wrote 4H features -> %s (%.2f MB)", path, path.stat().st_size / 1e6)


# =============================================================================
# 3) Train
# =============================================================================

def time_series_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    n = len(df)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL].astype("int8")
    X_val, y_val = val[FEATURE_COLS], val[TARGET_COL].astype("int8")
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL].astype("int8")
    log.info("Split: train=%s  val=%s  test=%s", X_train.shape, X_val.shape, X_test.shape)
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_lightgbm(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
) -> lgb.Booster:
    log.info("[3/8] Training LightGBM with early stopping (patience=%d)", EARLY_STOPPING)
    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, free_raw_data=False)
    t0 = time.time()
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
    log.info("Training done in %.1fs. Best iter=%d, best val_logloss=%.6f",
             time.time() - t0, booster.best_iteration, booster.best_score["val"]["binary_logloss"])
    return booster


def evaluate_test(booster: lgb.Booster, X_test: pd.DataFrame, y_test: pd.Series) -> np.ndarray:
    p = booster.predict(X_test, num_iteration=booster.best_iteration)
    y_pred = (p > 0.5).astype("int8")
    print("\n=== Test Classification Report (4H, threshold=0.5) ===")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("Confusion matrix:")
    print(pd.DataFrame(cm, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]))
    return p  # return probabilities for the backtest


def save_model(booster: lgb.Booster, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))
    log.info("Saved model -> %s", path)


# =============================================================================
# 4) Backtest V3
# =============================================================================

def build_trade_log(
    test: pd.DataFrame, p_up: np.ndarray, threshold: float
) -> pd.DataFrame:
    """Long-only, 1-bar holding (4h). No overlap, no leverage."""
    close = test["Close"].to_numpy()
    times = test.index.to_numpy()
    candidates = np.flatnonzero((p_up > threshold) & np.isfinite(close))
    if candidates.size == 0:
        return pd.DataFrame(columns=[
            "entry_time", "exit_time", "entry_price", "exit_price",
            "ret_gross", "ret_net", "p_up",
        ])

    last_exit = -1
    entries: List[int] = []
    for c in candidates:
        if c <= last_exit:
            continue
        if c + TARGET_HORIZON >= len(close):
            break
        entries.append(int(c))
        last_exit = c + TARGET_HORIZON

    entry_idx = np.asarray(entries, dtype=np.int64)
    exit_idx = entry_idx + TARGET_HORIZON
    entry_px = close[entry_idx]
    exit_px = close[exit_idx]
    ret_gross = exit_px / entry_px - 1.0
    ret_net = ret_gross - FEE_ROUND_TRIP
    return pd.DataFrame({
        "entry_time": pd.to_datetime(times[entry_idx]),
        "exit_time": pd.to_datetime(times[exit_idx]),
        "entry_price": entry_px,
        "exit_price": exit_px,
        "ret_gross": ret_gross,
        "ret_net": ret_net,
        "p_up": p_up[entry_idx].astype("float32"),
    })


def build_equity_curve_fixed_fractional(
    test: pd.DataFrame, trades: pd.DataFrame,
    initial_capital: float, risk_fraction: float,
) -> pd.Series:
    """
    Equity curve on the 4H bar grid. Each trade holds exactly one bar, so
    the equity is a step function that steps at each trade's exit_time by
    (1 + risk_fraction * ret_net).  We avoid any int-comparison trickery
    (the test index is microsecond-resolution while Timestamp.value is
    nanosecond, which caused a unit mismatch in the first version) and
    instead build a sparse per-exit equity series, then reindex+ffill into
    the full 4H grid.
    """
    equity = pd.Series(initial_capital, index=test.index, dtype="float64")
    if trades.empty:
        return equity

    growth = 1.0 + risk_fraction * trades["ret_net"].to_numpy()
    cum_growth = np.cumprod(growth)
    exit_equity = initial_capital * cum_growth

    # Sparse series: one equity value at each trade's exit time.
    sparse = pd.Series(exit_equity, index=trades["exit_time"].to_numpy())
    # Reindex onto the dense 4H grid; forward-fill between trades.
    dense = sparse.reindex(test.index).ffill()
    # The first few bars before any trade should stay at initial_capital.
    equity[:] = dense.fillna(initial_capital).to_numpy()
    return equity


def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict:
    final = float(equity.iloc[-1])
    roi = final / INITIAL_CAPITAL - 1.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min())

    eq_ret = equity.pct_change().fillna(0.0)
    mean_per = float(eq_ret.mean())
    std_per = float(eq_ret.std(ddof=0))
    # 6 bars per day at 4H, 365 days -> 6*365 = 2190 bars/year
    periods_per_year = (24 // 4) * 365
    sharpe = (mean_per / std_per) * math.sqrt(periods_per_year) if std_per > 0 else float("nan")

    n_trades = int(len(trades))
    win_rate = float((trades["ret_net"] > 0).mean()) if n_trades else float("nan")
    avg_trade = float(trades["ret_net"].mean()) if n_trades else float("nan")
    return {
        "final_equity": final,
        "roi_pct": roi * 100.0,
        "max_drawdown_pct": max_dd * 100.0,
        "annualized_sharpe": sharpe,
        "n_trades": n_trades,
        "win_rate_pct": win_rate * 100.0 if n_trades else float("nan"),
        "avg_trade_ret_pct": avg_trade * 100.0 if n_trades else float("nan"),
    }


def plot_equity(equity: pd.Series, metrics: dict, threshold: float) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(equity.index, equity.values, color="#1f77b4", linewidth=1.3, label="4H strategy equity")
    ax.axhline(INITIAL_CAPITAL, color="#888888", linestyle="--", linewidth=0.9, label="Initial capital")
    ax.set_title(
        f"BTC/USD 4H LightGBM V3  |  thr={threshold:.2f}  |  "
        f"ROI {metrics['roi_pct']:+.2f}%  |  "
        f"Sharpe {metrics['annualized_sharpe']:.2f}  |  "
        f"MaxDD {metrics['max_drawdown_pct']:.2f}%"
    )
    ax.set_ylabel("Equity (USD)")
    ax.set_xlabel("Time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(EQUITY_PNG, dpi=140)
    plt.close(fig)
    log.info("Saved equity curve -> %s", EQUITY_PNG)


# =============================================================================
# Orchestration
# =============================================================================

def run() -> None:
    bars = resample_to_4h(CLEAN_PATH)
    feats = build_features_4h(bars)
    save_features(feats, FEATURES_4H_PATH)

    X_train, X_val, X_test, y_train, y_val, y_test = time_series_split(feats)
    booster = train_lightgbm(X_train, y_train, X_val, y_val)
    print(f"[4/8] Best iter: {booster.best_iteration}")

    p_test = evaluate_test(booster, X_test, y_test)
    save_model(booster, MODEL_PATH)

    # Build a test-slice frame with Close + the same DatetimeIndex used for backtest.
    # `X_test` and `y_test` already share the index from the split; pull Close back.
    test_full = feats.iloc[len(feats) - len(X_test):]
    test_full = test_full.assign(p_up=p_test.astype("float32"))

    n_long = int((p_test > PROB_THRESHOLD).sum())
    print(f"\n[5/8] Test period: {len(X_test):,} bars  "
          f"[{test_full.index.min()} .. {test_full.index.max()}]")
    print(f"        P(up) > {PROB_THRESHOLD:.2f}  -> {n_long:,} candidate bars  "
          f"(of {len(X_test):,})")
    print(f"        P(up) range on test: min={p_test.min():.4f}, "
          f"max={p_test.max():.4f}, mean={p_test.mean():.4f}")

    trades = build_trade_log(test_full, p_test, PROB_THRESHOLD)
    print(f"[6/8] Trade log rows: {len(trades):,}")
    if not trades.empty:
        print(f"        mean ret_net = {trades['ret_net'].mean():.6f}, "
              f"std = {trades['ret_net'].std():.6f}, "
              f"min = {trades['ret_net'].min():.6f}, max = {trades['ret_net'].max():.6f}")
        print(f"        avg p_up at entry = {trades['p_up'].mean():.4f}")

    equity = build_equity_curve_fixed_fractional(
        test_full, trades, INITIAL_CAPITAL, RISK_FRACTION
    )
    metrics = compute_metrics(equity, trades)

    print("\n=== Backtest V3 (4H, Out-of-Sample Test) ===")
    print(f"  Threshold         : P(up) > {PROB_THRESHOLD:.2f}")
    print(f"  Position sizing   : {int(RISK_FRACTION*100)}% of equity per trade")
    print(f"  Initial capital   : ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final equity      : ${metrics['final_equity']:,.2f}")
    print(f"  ROI               : {metrics['roi_pct']:+.2f} %")
    print(f"  Max drawdown      : {metrics['max_drawdown_pct']:.2f} %")
    print(f"  Annualized Sharpe : {metrics['annualized_sharpe']:.2f}")
    print(f"  Total trades      : {metrics['n_trades']:,}")
    print(f"  Win rate          : {metrics['win_rate_pct']:.2f} %")
    print(f"  Avg trade ret     : {metrics['avg_trade_ret_pct']:+.4f} %")

    plot_equity(equity, metrics, PROB_THRESHOLD)
    print(f"[8/8] Saved equity plot -> {EQUITY_PNG}")


if __name__ == "__main__":
    run()
