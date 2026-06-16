"""
BTC/USD 1-minute data preprocessing and cleaning.

Pipeline:
    1. Load raw CSV with memory-efficient dtypes; keep Timestamp as int64.
    2. Convert Unix seconds -> tz-naive UTC DatetimeIndex and set as index.
    3. Reindex to a continuous 1-minute grid (min -> max) so missing minutes
       become real NaN rows. (Critical: do NOT drop gaps, or technical
       indicators later will be mis-aligned in time.)
    4. Impute with financial-market logic:
         - Close:  ffill (last traded price persists when no trade occurs).
         - Open/High/Low: inherit the ffill-ed Close of the same minute.
         - Volume: 0.0 (no trade -> no volume).
    5. Persist cleaned dataset to data/processed/btcusd_cleaned.csv.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd


# ---------- Configuration -----------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_PATH: Path = PROJECT_ROOT / "data" / "raw" / "btcusd_1-min_data.csv"
CLEAN_PATH: Path = PROJECT_ROOT / "data" / "processed" / "btcusd_cleaned.csv"

# float32 halves memory vs. float64 with negligible loss for OHLCV.
RAW_DTYPES: dict = {
    "Open": "float32",
    "High": "float32",
    "Low": "float32",
    "Close": "float32",
    "Volume": "float32",
}
DATETIME_COL: str = "Timestamp"
FREQ: str = "1min"
PRICE_COLS: Tuple[str, ...] = ("Open", "High", "Low", "Close")
VOLUME_COL: str = "Volume"


# ---------- Logging -----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("preprocess_btcusd")


# ---------- Pipeline steps ----------------------------------------------------

def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw CSV with memory-efficient dtypes; keep Timestamp as int64."""
    log.info("Loading raw data from %s", path)
    df = pd.read_csv(
        path,
        dtype=RAW_DTYPES,  # Timestamp defaults to int64; converted below.
    )
    log.info(
        "Raw data loaded with shape %s (memory: %.2f MB)",
        df.shape,
        df.memory_usage(deep=True).sum() / 1e6,
    )
    return df


def to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Unix seconds to a tz-naive UTC DatetimeIndex and sort it."""
    idx = pd.DatetimeIndex(pd.to_datetime(df[DATETIME_COL], unit="s", utc=True)).tz_convert(None)
    df = df.drop(columns=[DATETIME_COL])
    df.index = idx
    df = df.sort_index()
    if df.index.has_duplicates:
        n_dup = int(df.index.duplicated().sum())
        log.warning("Dropping %d duplicate timestamps before reindex.", n_dup)
        df = df[~df.index.duplicated(keep="first")]
    return df


def reindex_to_continuous_1min(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Reindex to a strict 1-minute grid; return df and the number of inserted rows."""
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=FREQ)
    original_rows = len(df)
    df = df.reindex(full_index)
    inserted_rows = len(df) - original_rows
    log.info(
        "Reindexed to %s grid: original rows=%d, new rows=%d, gaps filled=%d",
        FREQ, original_rows, len(df), inserted_rows,
    )
    return df, inserted_rows


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Apply financial-logic imputation rules on a defensive copy."""
    df = df.copy()

    # 1) Close: price holds when no new trade arrives.
    df["Close"] = df["Close"].ffill()

    # 2) Open / High / Low: inherit the (now ffill-ed) Close of the same minute.
    for col in ("Open", "High", "Low"):
        df[col] = df[col].fillna(df["Close"])

    # 3) Volume: 0.0 when no trade occurred.
    df[VOLUME_COL] = df[VOLUME_COL].fillna(0.0)

    # Safety net: if the very first row is missing (no prior Close to ffill),
    # back-fill once. Should not happen in practice on this dataset.
    if df[list(PRICE_COLS)].iloc[0].isna().any():
        log.warning("Leading NaNs detected; applying bfill as a safety net.")
        df[list(PRICE_COLS)] = df[list(PRICE_COLS)].bfill()

    return df


def save_clean(df: pd.DataFrame, path: Path) -> None:
    """Persist the cleaned frame; index is the DatetimeIndex (ISO-8601)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True, index_label="Timestamp", float_format="%.6f")
    size_mb = path.stat().st_size / 1e6
    log.info("Cleaned data written to %s (%.2f MB)", path, size_mb)


# ---------- Orchestration -----------------------------------------------------

def run() -> None:
    raw = load_raw(RAW_PATH)
    print(f"[1/4] Raw data shape:                {raw.shape}")

    raw = to_datetime_index(raw)
    continuous, gaps = reindex_to_continuous_1min(raw)
    print(f"[2/4] After reindex to 1-min grid:   {continuous.shape}")
    print(f"      Time gaps successfully filled: {gaps:,}")

    cleaned = impute_missing(continuous)
    print(f"[3/4] After imputation:              {cleaned.shape}")

    assert not cleaned[list(PRICE_COLS)].isna().any().any(), "NaNs remain in price columns."
    assert not cleaned[VOLUME_COL].isna().any(), "NaNs remain in Volume."
    assert len(cleaned) == len(continuous), "Row count changed during imputation."

    save_clean(cleaned, CLEAN_PATH)
    print(f"[4/4] Saved cleaned CSV -> {CLEAN_PATH}")

    print("\n--- Cleaned dataset summary ---")
    print(f"Range: {cleaned.index.min()}  ->  {cleaned.index.max()}")
    print(f"Rows : {len(cleaned):,}")
    print(cleaned[list(PRICE_COLS) + [VOLUME_COL]].describe().round(4))


if __name__ == "__main__":
    run()
