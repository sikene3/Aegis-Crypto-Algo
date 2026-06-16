"""
BTC/USD 15-minute LightGBM V2 backtester (probability-threshold gated).

V1 took every model-1 signal and compounded them geometrically into an
account-annihilating result (-100%) because the per-trade edge was
smaller than the round-trip fee and we ran ~44k trades.

V2 changes two things:
    1. Signal gate: only enter when P(up) > THRESHOLD (default 0.80).
       This cuts the signal count by an order of magnitude and keeps
       only the highest-conviction setups where the fee can be absorbed.
    2. Position sizing: risk a fixed fraction of current equity per trade
       (RISK_FRACTION, default 0.01 = 1%). This bounds the geometric
       blow-up from V1 and gives a realistic equity path.

If the user-specified threshold yields zero signals on this test slice
(the model's calibrated probabilities are tight: max P(up) ~ 0.56 on
this OOS window), we auto-relax to the 99th percentile of P(up) so the
report still has a meaningful trade log. The relaxed threshold is
reported transparently in the output.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List

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
EQUITY_PNG: Path = OUTPUTS_DIR / "portfolio_growth_v2.png"

# Sequential split (must match train_lgbm.py / backtest_lgbm.py)
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
FEE_PER_SIDE: float = 0.001
FEE_ROUND_TRIP: float = 2.0 * FEE_PER_SIDE

# V2 knobs
PROB_THRESHOLD: float = 0.80         # user-specified gate
PROB_FALLBACK_QUANTILE: float = 0.99  # auto-relax target if 0 signals
RISK_FRACTION: float = 0.01          # 1% of equity per trade (fixed fractional)
MINUTES_PER_YEAR: int = 60 * 24 * 365


# ---------- Logging -----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("backtest_lgbm_v2")


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
    n = len(df)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    test = df.iloc[n_train + n_val:]
    log.info("Test slice: %s  [%s .. %s]", test.shape, test.index.min(), test.index.max())
    return test


def choose_threshold(p_up: np.ndarray, requested: float) -> tuple[float, str]:
    """Apply the requested threshold; fall back to a percentile if it's empty."""
    n_req = int((p_up > requested).sum())
    if n_req >= 10:
        return requested, "user-specified"
    fallback = float(np.quantile(p_up, PROB_FALLBACK_QUANTILE))
    log.warning(
        "PROB_THRESHOLD=%.2f yielded %d signals; auto-relaxing to p%.0f(P_up)=%.4f",
        requested, n_req, PROB_FALLBACK_QUANTILE * 100, fallback,
    )
    return fallback, f"p{int(PROB_FALLBACK_QUANTILE*100)} fallback"


# ---------- Signal -> trades -------------------------------------------------

def build_trade_log(
    test: pd.DataFrame, p_up: np.ndarray, threshold: float
) -> pd.DataFrame:
    """
    Filter on P(up) > threshold, then walk forward, skipping overlapping
    entries, to build a per-trade log with the model confidence at entry.
    """
    close = test["Close"].to_numpy()
    times = test.index.to_numpy()
    candidates = np.flatnonzero((p_up > threshold) & np.isfinite(close))
    if candidates.size == 0:
        return pd.DataFrame(columns=[
            "entry_time", "exit_time",
            "entry_price", "exit_price",
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


# ---------- Equity curve + metrics -------------------------------------------

def build_equity_curve_fixed_fractional(
    test: pd.DataFrame, trades: pd.DataFrame,
    initial_capital: float, risk_fraction: float,
) -> pd.Series:
    """Fixed-fractional equity curve: per-trade PnL = risk_fraction * equity * ret_net."""
    equity = pd.Series(initial_capital, index=test.index, dtype="float64")
    if trades.empty:
        return equity

    entry_ns = trades["entry_time"].to_numpy().astype("datetime64[ns]").view("i8")
    exit_ns = trades["exit_time"].to_numpy().astype("datetime64[ns]").view("i8")
    ret_net = trades["ret_net"].to_numpy()
    test_ns = test.index.asi8

    growth = 1.0 + risk_fraction * ret_net
    cum_growth = np.concatenate(([1.0], np.cumprod(growth)))

    pos = np.searchsorted(exit_ns, test_ns, side="right") - 1
    has_closed = pos >= 0

    closed_step = np.ones_like(test_ns, dtype="float64")
    closed_step[has_closed] = cum_growth[pos[has_closed]]

    close_arr = test["Close"].to_numpy()
    in_pos_idx = np.searchsorted(entry_ns, test_ns, side="right") - 1
    ep = trades["entry_price"].to_numpy()
    in_pos_idx_safe = np.where(in_pos_idx >= 0, in_pos_idx, 0)
    inside = (
        (in_pos_idx >= 0)
        & np.greater(exit_ns[np.clip(in_pos_idx_safe, 0, len(exit_ns) - 1)], test_ns)
    )
    mtm = np.where(
        inside,
        close_arr / np.where(inside, ep[np.clip(in_pos_idx_safe, 0, len(ep) - 1)], 1.0) - 1.0,
        0.0,
    )
    ramp = 1.0 + risk_fraction * mtm

    equity_arr = initial_capital * closed_step * ramp
    equity[:] = equity_arr
    return equity


def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict:
    final = float(equity.iloc[-1])
    roi = final / INITIAL_CAPITAL - 1.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min())

    eq_ret = equity.pct_change().fillna(0.0)
    mean_min = float(eq_ret.mean())
    std_min = float(eq_ret.std(ddof=0))
    sharpe = (mean_min / std_min) * math.sqrt(MINUTES_PER_YEAR) if std_min > 0 else float("nan")

    n_trades = int(len(trades))
    win_rate = float((trades["ret_net"] > 0).mean()) if n_trades else float("nan")
    avg_trade = float(trades["ret_net"].mean()) if n_trades else float("nan")
    avg_winner = float(trades.loc[trades["ret_net"] > 0, "ret_net"].mean()) if (trades["ret_net"] > 0).any() else float("nan")
    avg_loser = float(trades.loc[trades["ret_net"] <= 0, "ret_net"].mean()) if (trades["ret_net"] <= 0).any() else float("nan")
    return {
        "final_equity": final,
        "roi_pct": roi * 100.0,
        "max_drawdown_pct": max_dd * 100.0,
        "annualized_sharpe": sharpe,
        "n_trades": n_trades,
        "win_rate_pct": win_rate * 100.0 if n_trades else float("nan"),
        "avg_trade_ret_pct": avg_trade * 100.0 if n_trades else float("nan"),
        "avg_winner_pct": avg_winner * 100.0 if n_trades else float("nan"),
        "avg_loser_pct": avg_loser * 100.0 if n_trades else float("nan"),
    }


# ---------- Plotting ---------------------------------------------------------

def plot_equity(equity: pd.Series, metrics: dict, threshold: float, source: str) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(equity.index, equity.values, color="#2ca02c", linewidth=1.3, label="V2 strategy equity")
    ax.axhline(INITIAL_CAPITAL, color="#888888", linestyle="--", linewidth=0.9, label="Initial capital")
    ax.set_title(
        f"BTC/USD 15m LightGBM V2  |  thr={threshold:.4f} ({source})  |  "
        f"size={int(RISK_FRACTION*100)}%  |  "
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

    p_up = booster.predict(test[FEATURE_COLS], num_iteration=booster.current_iteration())
    p_up = p_up.astype("float32")
    n_total = int((p_up > 0.5).sum())

    threshold, source = choose_threshold(p_up, PROB_THRESHOLD)
    n_v2 = int((p_up > threshold).sum())

    print(f"[1/5] Test slice: {test.shape}  "
          f"[{test.index.min()} .. {test.index.max()}]")
    print(f"[2/5] Probability distribution on test:")
    print(f"      mean P(up)    = {p_up.mean():.4f}, "
          f"min={p_up.min():.4f}, max={p_up.max():.4f}")
    print(f"      P(up) > 0.50            -> {n_total:,} signals  (V1 baseline)")
    print(f"      P(up) > {PROB_THRESHOLD:.2f}            -> "
          f"{int((p_up > PROB_THRESHOLD).sum()):,} signals  (user threshold)")
    print(f"      P(up) > {threshold:.4f} (active)  -> {n_v2:,} signals  ({source})")

    trades = build_trade_log(test, p_up, threshold)
    print(f"[3/5] Trade log rows: {len(trades):,}")
    if not trades.empty:
        print(f"      trade ret (net) -> "
              f"mean={trades['ret_net'].mean():.6f}, "
              f"std={trades['ret_net'].std():.6f}, "
              f"min={trades['ret_net'].min():.6f}, max={trades['ret_net'].max():.6f}")
        print(f"      avg p_up at entry = {trades['p_up'].mean():.4f}")

    equity = build_equity_curve_fixed_fractional(
        test, trades, INITIAL_CAPITAL, RISK_FRACTION
    )
    metrics = compute_metrics(equity, trades)

    print("\n=== Backtest Metrics V2 (Out-of-Sample Test) ===")
    print(f"  Active threshold  : P(up) > {threshold:.4f}  ({source})")
    print(f"  Position sizing   : {int(RISK_FRACTION*100)}% of equity per trade")
    print(f"  Initial capital   : ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final equity      : ${metrics['final_equity']:,.2f}")
    print(f"  ROI               : {metrics['roi_pct']:+.2f} %")
    print(f"  Max drawdown      : {metrics['max_drawdown_pct']:.2f} %")
    print(f"  Annualized Sharpe : {metrics['annualized_sharpe']:.2f}")
    print(f"  Trades executed   : {metrics['n_trades']:,}")
    print(f"  Win rate          : {metrics['win_rate_pct']:.2f} %")
    print(f"  Avg trade ret     : {metrics['avg_trade_ret_pct']:+.4f} %")
    print(f"  Avg winner        : {metrics['avg_winner_pct']:+.4f} %")
    print(f"  Avg loser         : {metrics['avg_loser_pct']:+.4f} %")

    plot_equity(equity, metrics, threshold, source)
    print(f"[5/5] Saved equity plot -> {EQUITY_PNG}")


if __name__ == "__main__":
    run()
