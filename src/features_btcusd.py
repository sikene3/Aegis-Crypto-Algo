"""
BTC/USD 1-minute feature engineering.

Reads the cleaned OHLCV dataset produced by `preprocess_btcusd.py` and adds
quantitative features plus a forward-looking binary target. All indicators
are computed with pure pandas/numpy vectorization (no TA-Lib dependency).

Indicators
----------
- log_return      : log(Close_t / Close_{t-1})
- SMA_50/200      : simple moving averages of Close
- EMA_9/21        : exponential moving averages of Close (Wilder-style adjust=False)
- RSI_14          : Wilder's Relative Strength Index
- MACD            : EMA12 - EMA26, with EMA9 signal line and histogram
- Bollinger Bands : 20-period SMA +/- 2*std on Close -> BB_upper, BB_lower

Target
------
- target_15m : 1 if Close_{t+15} > Close_t, else 0  (binary classification)

Output
------
- data/processed/btcusd_features.csv (NaN rows dropped)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# ---------- Configuration -----------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CLEAN_PATH: Path = PROJECT_ROOT / "data" / "processed" / "btcusd_cleaned.csv"
FEATURES_PATH: Path = PROJECT_ROOT / "data" / "processed" / "btcusd_features.csv"

# Indicator windows
SMA_FAST: int = 50
SMA_SLOW: int = 200
EMA_FAST: int = 9
EMA_MID: int = 21
RSI_PERIOD: int = 14
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
BB_PERIOD: int = 20
BB_STD: float = 2.0

# Target horizon (minutes ahead)
TARGET_HORIZON: int = 15


# ---------- Logging -----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("features_btcusd")


# ---------- IO helpers --------------------------------------------------------

def load_clean(path: Path) -> pd.DataFrame:
    """Load cleaned OHLCV with memory-aware dtypes; index -> DatetimeIndex."""
    log.info("Loading cleaned data from %s", path)
    df = pd.read_csv(
        path,
        index_col="Timestamp",
        parse_dates=["Timestamp"],
        dtype={
            "Open": "float32",
            "High": "float32",
            "Low": "float32",
            "Close": "float32",
            "Volume": "float32",
        },
    )
    log.info(
        "Cleaned data loaded with shape %s (memory: %.2f MB)",
        df.shape,
        df.memory_usage(deep=True).sum() / 1e6,
    )
    return df


def save_features(df: pd.DataFrame, path: Path) -> None:
    """Persist engineered features; index labeled 'Timestamp' (ISO-8601)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True, index_label="Timestamp", float_format="%.6f")
    size_mb = path.stat().st_size / 1e6
    log.info("Feature dataset written to %s (%.2f MB)", path, size_mb)


# ---------- Indicator builders -----------------------------------------------

def add_returns(df: pd.DataFrame) -> None:
    """Log return of Close: log(Close_t / Close_{t-1})."""
    close = df["Close"]
    # np.log is faster than df['Close'].apply(np.log) and avoids object dtype.
    df["log_return"] = np.log(close / close.shift(1))


def add_moving_averages(df: pd.DataFrame) -> None:
    """SMA and EMA of Close (adjust=False for true recursive EMA)."""
    close = df["Close"]
    df["SMA_50"] = close.rolling(window=SMA_FAST, min_periods=SMA_FAST).mean()
    df["SMA_200"] = close.rolling(window=SMA_SLOW, min_periods=SMA_SLOW).mean()
    df["EMA_9"] = close.ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA_21"] = close.ewm(span=EMA_MID, adjust=False).mean()


def add_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> None:
    """Wilder's RSI on Close. Uses ewm alpha=1/period, the standard formulation."""
    close = df["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing is an EMA with alpha = 1/period and adjust=False.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    df["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))


def add_macd(
    df: pd.DataFrame,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> None:
    """MACD line = EMA(fast) - EMA(slow); signal = EMA(MACD, signal); hist = MACD - signal."""
    close = df["Close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    df["MACD"] = macd_line
    df["MACD_signal"] = signal_line
    df["MACD_hist"] = macd_line - signal_line


def add_bollinger(df: pd.DataFrame, period: int = BB_PERIOD, std: float = BB_STD) -> None:
    """Bollinger Bands: SMA(period) +/- std * rolling std on Close."""
    close = df["Close"]
    mid = close.rolling(window=period, min_periods=period).mean()
    sd = close.rolling(window=period, min_periods=period).std(ddof=0)
    df["BB_mid"] = mid
    df["BB_upper"] = mid + std * sd
    df["BB_lower"] = mid - std * sd


def add_forward_target(df: pd.DataFrame, horizon: int = TARGET_HORIZON) -> None:
    """Binary target: 1 if Close_{t+horizon} > Close_t, else 0."""
    future_close = df["Close"].shift(-horizon)
    df["target_15m"] = (future_close > df["Close"]).astype("int8")


# ---------- Orchestration -----------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run every feature builder; return the enriched frame (still with NaNs)."""
    add_returns(df)
    add_moving_averages(df)
    add_rsi(df)
    add_macd(df)
    add_bollinger(df)
    add_forward_target(df)
    return df


def run() -> None:
    df = load_clean(CLEAN_PATH)
    print(f"[1/5] Loaded cleaned data shape:     {df.shape}")

    df = build_features(df)
    print(f"[2/5] After feature engineering:     {df.shape}  (with NaNs from rolling/shift)")

    # Drop warm-up NaNs (head) and target-shift NaNs (tail).
    n_before = len(df)
    df = df.dropna()
    n_after = len(df)
    dropped = n_before - n_after
    print(f"[3/5] After dropna():                {df.shape}  (dropped {dropped:,} rows)")

    # Class balance for the binary target.
    counts = df["target_15m"].value_counts().sort_index()
    pct = (counts / counts.sum() * 100).round(2)
    print("[4/5] target_15m class balance:")
    for label, c in counts.items():
        print(f"        class={label}: {c:,}  ({pct.loc[label]}%)")

    save_features(df, FEATURES_PATH)
    print(f"[5/5] Saved engineered features -> {FEATURES_PATH}")

    # Sanity: no NaNs should remain, and all expected columns are present.
    expected = {
        "log_return", "SMA_50", "SMA_200", "EMA_9", "EMA_21", "RSI_14",
        "MACD", "MACD_signal", "MACD_hist",
        "BB_mid", "BB_upper", "BB_lower", "target_15m",
    }
    missing = expected - set(df.columns)
    assert not missing, f"Missing expected feature columns: {sorted(missing)}"
    assert not df.isna().any().any(), "NaNs remain after dropna()."


if __name__ == "__main__":
    run()
