"""
BTC/USD 15-minute LightGBM signal backtester.

Loads the engineered features, recreates the strict sequential 70/15/15 split
from `train_lgbm.py` to isolate the out-of-sample test slice, reloads the
saved LightGBM booster, simulates a long-only 15-minute holding strategy
with 0.1% round-trip-equivalent fee, and prints financial metrics + an
equity curve PNG.

Trading rule
------------
- Predict 1 at minute t  ->  enter LONG at Close[t]
- Exit 15 minutes later   ->  exit  at Close[t+15]
- Profit on a trade      =  (Close[t+15] / Close[t] - 1) - 2 * FEE
  (the 2x captures entry + exit fees, conservative round-trip cost).
- Position sizing: fully invested for the duration of each open trade.
  This is a long-only, fully-collateralized backtest, not a margin model.
- We only ever have one position open at a time, so overlapping signals in
  the 15-minute holding window are ignored (next entry is taken only after
  the current trade closes).

All PnL math is done with pandas/numpy vectorization on a per-trade
dataframe; the equity curve is built by stepping a position-aware ledger
and is then resampled to a fixed 1-minute grid for plotting.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Tuple

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------- Configuration -----------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
FEATURES_PATH: Path = PROJECT_ROOT / "data" / "processed" / "btcusd_features.csv"
MODEL_PATH: Path = PROJECT_ROOT / "outputs" / "lgbm_model.txt"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
EQUITY_PNG: Path = OUTPUTS_DIR / "portfolio_growth.png"

# Sequential split (must match train_lgbm.py)
TRAIN_FRAC: float = 0.70
VAL_FRAC: float = 0.15
TEST_FRAC: float = 0.15
TARGET_HORIZON: int = 15
TARGET_COL: str = "target_15m"
FEATURE_COLS: List[str] = [
    "log_return", "SMA_50", "SMA_200", "EMA_9", "EMA_21",
    "RSI_14", "MACD", "MACD_signal", "MACD_hist",
    "BB_mid", "BB_upper", "BB_lower",
]

# Capital + cost model
INITIAL_CAPITAL: float = 10_000.0
FEE_PER_SIDE: float = 0.001        # 0.1% per side (entry + exit)
FEE_ROUND_TRIP: float = 2.0 * FEE_PER_SIDE
# Annualization: there are 525,600 minutes in a 365-day year.
MINUTES_PER_YEAR: int = 60 * 24 * 365


# ---------- Logging -----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("backtest_lgbm")


# ---------- IO + model -------------------------------------------------------

def load_features(path: Path) -> pd.DataFrame:
    log.info("Loading features from %s", path)
    df = pd.read_csv(
        path,
        index_col="Timestamp",
        parse_dates=["Timestamp"],
        dtype={c: "float32" for c in FEATURE_COLS},
    )
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])
    log.info("Features loaded: shape=%s, memory=%.2f MB",
             df.shape, df.memory_usage(deep=True).sum() / 1e6)
    return df


def load_model(path: Path) -> lgb.Booster:
    log.info("Loading LightGBM booster from %s", path)
    return lgb.Booster(model_file=str(path))


def isolate_test(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate the exact sequential 70/15/15 split and return the test slice."""
    n = len(df)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    test = df.iloc[n_train + n_val:]
    log.info("Test slice: %s  [%s .. %s]", test.shape, test.index.min(), test.index.max())
    return test


# ---------- Signal -> trades -------------------------------------------------

def build_trade_log(
    test: pd.DataFrame, y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Convert 0/1 predictions into a per-trade log with entry/exit timestamps
    and per-trade return. Strict no-overlap: we skip signals that would
    open a trade while another is still open.

    Returns a DataFrame with columns:
        entry_time, exit_time, entry_price, exit_price, ret_gross, ret_net
    """
    close = test["Close"].to_numpy()
    times = test.index.to_numpy()
    pred = y_pred.astype("int8").ravel()

    # Identify candidate entry indices (where model says 1 and Close is finite).
    candidates = np.flatnonzero((pred == 1) & np.isfinite(close))
    if candidates.size == 0:
        return pd.DataFrame(columns=[
            "entry_time", "exit_time",
            "entry_price", "exit_price",
            "ret_gross", "ret_net",
        ])

    # Walk forward, skipping overlapping signals.
    last_exit = -1
    entries: List[int] = []
    for c in candidates:
        if c <= last_exit:
            continue
        if c + TARGET_HORIZON >= len(close):
            break  # not enough bars to close the trade
        entries.append(int(c))
        last_exit = c + TARGET_HORIZON

    entry_idx = np.asarray(entries, dtype=np.int64)
    exit_idx = entry_idx + TARGET_HORIZON

    entry_px = close[entry_idx]
    exit_px = close[exit_idx]
    ret_gross = exit_px / entry_px - 1.0
    ret_net = ret_gross - FEE_ROUND_TRIP

    trades = pd.DataFrame({
        "entry_time": pd.to_datetime(times[entry_idx]),
        "exit_time": pd.to_datetime(times[exit_idx]),
        "entry_price": entry_px,
        "exit_price": exit_px,
        "ret_gross": ret_gross,
        "ret_net": ret_net,
    })
    return trades


# ---------- Equity curve + metrics -------------------------------------------

def build_equity_curve(test: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """
    Build a 1-minute equity curve aligned to the test index.

    For each minute t in the test set:
        - in_position  -> equity grows by the per-minute return of the
                          open trade (compounded proportionally to minutes held)
        - flat         -> equity stays constant
    All vectorized: pre-compute per-trade cumulative return, then assign it
    to the [entry_time, exit_time] minute grid via index alignment.
    """
    # Baseline: flat capital.
    equity = pd.Series(INITIAL_CAPITAL, index=test.index, dtype="float64")

    if trades.empty:
        return equity.to_frame("equity")

    # For each trade, multiply the equity forward by (1 + ret_net) at the
    # exit timestamp. We use a "step" approach: equity stays at base during
    # the trade, then jumps at exit. This is conservative and matches the
    # cash-and-carry interpretation: you put capital in, get it back +/- PnL.
    # For a more realistic equity curve, we instead mark-to-market minute
    # by minute using the realized path of the close price.
    #
    # Mark-to-market implementation: for every minute t in the test range,
    # we ask "is t inside any open trade window?" via searchsorted on the
    # sorted entry/exit arrays. That gives O(n + m) per step.

    entry_times = trades["entry_time"].to_numpy()
    exit_times = trades["exit_time"].to_numpy()
    entry_px = trades["entry_price"].to_numpy()
    exit_px = trades["exit_price"].to_numpy()
    ret_net = trades["ret_net"].to_numpy()

    # Convert equity index to int64 ns for fast comparison.
    test_ns = test.index.asi8

    # We build the curve as: base * (1 + ret_net) once a trade closes,
    # ramping linearly from entry to exit using the realized price path
    # (Close-to-Close of the underlying). The ramp captures the within-
    # trade drawdown; the final mark uses ret_net (after fees).
    #
    # Because the test set has ~1.14M rows and we have ~hundreds of
    # thousands of trades in the long-only 1m signal case, the simplest
    # correct vectorized build is:
    #   for each minute t, find the most recent open trade that hasn't
    #   exited yet and mark-to-market.
    # We do that with np.searchsorted on the sorted exit_times array.
    #
    # equity[t] = INITIAL_CAPITAL * prod(1 + ret_net[j]) for all closed trades j up to t
    #           * (Close[t] / entry_price_of_open_trade - 0)   if currently in a trade
    # We approximate the in-trade ramp with the close-to-entry ratio
    # and apply fees on the mark-to-market portion to keep the curve smooth.
    #
    # This is fully vectorized via np.searchsorted.

    # 1) Closed-trade step component: cumulative product of (1+ret_net) over
    #    time, advanced at each exit_time.
    closed_step = np.empty(len(test_ns), dtype="float64")
    closed_step.fill(1.0)
    # exit_times are sorted by construction (entries are appended in order).
    # We need the position of the largest exit_time <= t for each t.
    # searchsorted(side='right') - 1 gives the index of the last exit <= t,
    # or -1 if none.
    exit_ns = exit_times.view("i8").astype(np.int64)
    pos = np.searchsorted(exit_ns, test_ns, side="right") - 1
    has_closed = pos >= 0
    cum_ret = np.concatenate(([1.0], np.cumprod(1.0 + ret_net)))
    closed_step[has_closed] = cum_ret[pos[has_closed]]

    # 2) Open-trade ramp: ratio of current Close to the most recent entry
    #    price, for those minutes that fall inside an open trade window.
    entry_ns = entry_times.view("i8").astype(np.int64)
    close_arr = test["Close"].to_numpy()
    # For each t, find the index of the most recent entry_time <= t.
    in_pos_idx = np.searchsorted(entry_ns, test_ns, side="right") - 1
    inside = (in_pos_idx >= 0) & np.greater(exit_ns[np.clip(in_pos_idx, 0, len(exit_ns)-1)], test_ns)
    # Mark-to-market: scale the closed-step equity by the within-trade ratio
    # of Close to entry price. No fee on the intermediate marks; fees are
    # applied at exit via the closed-step product.
    in_pos_idx_safe = np.where(inside, in_pos_idx, 0)
    open_ratio = np.where(
        inside,
        close_arr / np.where(inside, entry_px[np.clip(in_pos_idx_safe, 0, len(entry_px)-1)], 1.0),
        1.0,
    )

    equity_arr = INITIAL_CAPITAL * closed_step * open_ratio
    equity[:] = equity_arr
    return equity.to_frame("equity")


def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict:
    """Compute ROI, max drawdown, annualized Sharpe, and trade-level stats."""
    final = float(equity.iloc[-1])
    roi = final / INITIAL_CAPITAL - 1.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min())  # negative or zero

    # Per-minute returns of the equity curve (used for Sharpe).
    eq_ret = equity.pct_change().fillna(0.0)
    mean_min = float(eq_ret.mean())
    std_min = float(eq_ret.std(ddof=0))
    sharpe = (mean_min / std_min) * math.sqrt(MINUTES_PER_YEAR) if std_min > 0 else float("nan")

    n_trades = int(len(trades))
    win_rate = float((trades["ret_net"] > 0).mean()) if n_trades else float("nan")
    avg_trade = float(trades["ret_net"].mean()) if n_trades else float("nan")
    return {
        "final_equity": final,
        "roi_pct": roi * 100.0,
        "max_drawdown_pct": max_dd * 100.0,
        "annualized_sharpe": sharpe,
        "n_trades": n_trades,
        "win_rate_pct": win_rate * 100.0 if n_trades == n_trades else float("nan"),
        "avg_trade_ret_pct": avg_trade * 100.0 if n_trades == n_trades else float("nan"),
    }


# ---------- Plotting ---------------------------------------------------------

def plot_equity(equity: pd.Series, metrics: dict) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(equity.index, equity.values, color="#1f77b4", linewidth=1.2, label="Strategy equity")
    ax.axhline(INITIAL_CAPITAL, color="#888888", linestyle="--", linewidth=0.9, label="Initial capital")
    ax.set_title(
        f"BTC/USD 15m LightGBM Long-Only Strategy  |  "
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


# ---------- Orchestration ----------------------------------------------------

def run() -> None:
    df = load_features(FEATURES_PATH)
    test = isolate_test(df)
    booster = load_model(MODEL_PATH)
    log.info("Booster best_iter=%d, n_features=%d",
             booster.current_iteration(), booster.num_feature())

    print(f"[1/4] Test slice: {test.shape}  "
          f"[{test.index.min()} .. {test.index.max()}]")

    y_proba = booster.predict(test[FEATURE_COLS], num_iteration=booster.current_iteration())
    y_pred = (y_proba > 0.5).astype("int8")
    print(f"[2/4] Test signals: "
          f"longs={int(y_pred.sum()):,}  flat={int((y_pred == 0).sum()):,}")

    trades = build_trade_log(test, y_pred)
    print(f"      Trade log rows: {len(trades):,}")
    if not trades.empty:
        print(f"      Trade ret (net) -> mean={trades['ret_net'].mean():.6f}, "
              f"std={trades['ret_net'].std():.6f}, "
              f"min={trades['ret_net'].min():.6f}, max={trades['ret_net'].max():.6f}")

    equity = build_equity_curve(test, trades)
    metrics = compute_metrics(equity.iloc[:, 0], trades)

    print("\n=== Backtest Metrics (Out-of-Sample Test) ===")
    print(f"  Initial capital : ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final equity    : ${metrics['final_equity']:,.2f}")
    print(f"  ROI             : {metrics['roi_pct']:+.2f} %")
    print(f"  Max drawdown    : {metrics['max_drawdown_pct']:.2f} %")
    print(f"  Annualized Sharpe: {metrics['annualized_sharpe']:.2f}")
    print(f"  Trades executed : {metrics['n_trades']:,}")
    print(f"  Win rate        : {metrics['win_rate_pct']:.2f} %")
    print(f"  Avg trade ret   : {metrics['avg_trade_ret_pct']:+.4f} %")

    plot_equity(equity.iloc[:, 0], metrics)
    print(f"[4/4] Saved equity plot -> {EQUITY_PNG}")


if __name__ == "__main__":
    run()
