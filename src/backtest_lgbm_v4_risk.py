"""
BTC/USD 4-Hour V4 risk-managed backtester.

V3 compounded 100% of equity per trade with no SL/TP, so 329 trades
ending at -16 bps mean ret each fully participated in the loss and
annihilated the account (-42.94% ROI, -43.86% max drawdown).

V4 fixes that with production-grade risk management:
    * 10% position sizing (90% stays in cash).
    * Per-trade Stop Loss at -2.0% of entry.
    * Per-trade Take Profit at +4.0% of entry.
    * Hard time exit at the 1-bar (4H) mark if neither SL nor TP fires.
    * Mark-to-market against the bar's High/Low to decide SL/TP hits;
      if both fire on the same bar (gap through), use worst-case
      (conservative: stop out at SL price).
    * Fees 0.1% per side, applied on entry and on exit (round-trip 0.2%).

The 4H bars were resampled in `pipeline_4h.py` with O=first, H=max,
L=min, C=last, so within a single bar we can only tell whether the
SL or TP was *touched* (via High/Low), not the exact intra-bar order.
We follow the conservative convention: if both are touched, take SL.
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
FEATURES_PATH: Path = PROJECT_ROOT / "data" / "processed" / "btcusd_features_4h.csv"
MODEL_PATH: Path = PROJECT_ROOT / "outputs" / "lgbm_model_4h.txt"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
EQUITY_PNG: Path = OUTPUTS_DIR / "portfolio_growth_v4_risk_managed.png"

# Sequential split (must match pipeline_4h.py)
TRAIN_FRAC: float = 0.70
VAL_FRAC: float = 0.15
TEST_FRAC: float = 0.15
TARGET_HORIZON: int = 1
TARGET_COL: str = "target_1"
FEATURE_COLS: List[str] = [
    "log_return", "SMA_50", "SMA_200", "RSI_14",
    "MACD", "MACD_signal", "MACD_hist", "BB_upper", "BB_lower",
]

# Capital + risk
INITIAL_CAPITAL: float = 10_000.0
FEE_PER_SIDE: float = 0.001
FEE_ROUND_TRIP: float = 2.0 * FEE_PER_SIDE
RISK_FRACTION: float = 0.10         # 10% of equity per trade; 90% stays in cash
STOP_LOSS_PCT: float = 0.02         # -2.0%
TAKE_PROFIT_PCT: float = 0.04      # +4.0%
PROB_THRESHOLD: float = 0.55       # P(up) > 0.55  =>  long

# Annualization (4H bars -> 6 bars/day, 365 days)
BARS_PER_YEAR: int = 6 * 365


# ---------- Logging -----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("backtest_lgbm_v4_risk")


# ---------- IO + model -------------------------------------------------------

def load_features(path: Path) -> pd.DataFrame:
    log.info("Loading 4H features from %s", path)
    df = pd.read_csv(
        path,
        index_col="Timestamp",
        parse_dates=["Timestamp"],
        dtype={c: "float32" for c in FEATURE_COLS + ["Open", "High", "Low", "Close"]},
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


# ---------- Signal -> trades -------------------------------------------------

def build_trade_log(
    test: pd.DataFrame, p_up: np.ndarray, threshold: float
) -> pd.DataFrame:
    """
    Convert filtered P(up) into a per-trade log. Each trade knows:
        - entry / exit timestamps
        - entry / exit prices (exit price is whichever of SL/TP/close fires)
        - gross and net-of-fee return
        - exit reason (TP / SL / TIME)
    No overlapping trades.
    """
    close = test["Close"].to_numpy()
    high = test["High"].to_numpy()
    low = test["Low"].to_numpy()
    times = test.index.to_numpy()

    candidates = np.flatnonzero((p_up > threshold) & np.isfinite(close))
    if candidates.size == 0:
        return pd.DataFrame(columns=[
            "entry_time", "exit_time", "entry_price", "exit_price",
            "ret_gross", "ret_net", "p_up", "exit_reason",
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

    sl_px = entry_px * (1.0 - STOP_LOSS_PCT)
    tp_px = entry_px * (1.0 + TAKE_PROFIT_PCT)

    # Check SL / TP against the exit bar's High/Low.
    bar_high = high[exit_idx]
    bar_low = low[exit_idx]
    bar_close = close[exit_idx]

    hits_tp = bar_high >= tp_px
    hits_sl = bar_low <= sl_px
    # Conservative: if both, take SL.  Otherwise pick whichever was hit,
    # or fall back to time-exit at Close.
    exit_px = np.where(
        hits_sl, sl_px,
        np.where(hits_tp, tp_px, bar_close),
    )
    exit_reason = np.where(
        hits_sl, "SL",
        np.where(hits_tp, "TP", "TIME"),
    )

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
        "exit_reason": exit_reason,
    })


# ---------- Equity curve (cash + invested) -----------------------------------

def build_equity_curve(
    test: pd.DataFrame, trades: pd.DataFrame,
    initial_capital: float, risk_fraction: float,
) -> pd.Series:
    """
    Cash + invested ledger on the 4H bar grid.

    For each trade, `risk_fraction` of equity is deployed as position
    notional.  The 1-bar hold means the position is closed at the next
    bar, so the equity on the EXIT bar becomes:
        equity_exit = equity_at_entry * (1 + risk_fraction * ret_net)
    where ret_net already has the round-trip fee baked in.

    Between trades the equity is flat (cash earns nothing).
    """
    equity = pd.Series(initial_capital, index=test.index, dtype="float64")
    if trades.empty:
        return equity

    # We need the equity at the entry bar to size the position.  Because
    # positions are closed at the next bar, the equity path is:
    #   e[t+1] = e[t] * (1 + risk_fraction * ret_net)
    # where t is the entry bar index and t+1 is the exit bar index.
    growth_per_trade = 1.0 + risk_fraction * trades["ret_net"].to_numpy()
    cum_growth = np.cumprod(growth_per_trade)

    # Sparse series: equity at each trade's exit bar.
    exit_equity = initial_capital * cum_growth
    sparse = pd.Series(exit_equity, index=trades["exit_time"].to_numpy())

    # Reindex onto the dense 4H grid, forward-fill flat between trades.
    dense = sparse.reindex(test.index).ffill()
    equity[:] = dense.fillna(initial_capital).to_numpy()
    return equity


# ---------- Metrics + plot ---------------------------------------------------

def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict:
    final = float(equity.iloc[-1])
    roi = final / INITIAL_CAPITAL - 1.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min())

    eq_ret = equity.pct_change().fillna(0.0)
    mean_per = float(eq_ret.mean())
    std_per = float(eq_ret.std(ddof=0))
    sharpe = (mean_per / std_per) * math.sqrt(BARS_PER_YEAR) if std_per > 0 else float("nan")

    n_trades = int(len(trades))
    win_rate = float((trades["ret_net"] > 0).mean()) if n_trades else float("nan")
    avg_trade = float(trades["ret_net"].mean()) if n_trades else float("nan")
    n_tp = int((trades["exit_reason"] == "TP").sum()) if n_trades else 0
    n_sl = int((trades["exit_reason"] == "SL").sum()) if n_trades else 0
    n_time = int((trades["exit_reason"] == "TIME").sum()) if n_trades else 0
    return {
        "final_equity": final,
        "roi_pct": roi * 100.0,
        "max_drawdown_pct": max_dd * 100.0,
        "annualized_sharpe": sharpe,
        "n_trades": n_trades,
        "win_rate_pct": win_rate * 100.0 if n_trades else float("nan"),
        "avg_trade_ret_pct": avg_trade * 100.0 if n_trades else float("nan"),
        "n_tp": n_tp, "n_sl": n_sl, "n_time": n_time,
    }


def plot_equity(equity: pd.Series, metrics: dict) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(equity.index, equity.values, color="#1f77b4", linewidth=1.3,
            label="V4 risk-managed equity")
    ax.axhline(INITIAL_CAPITAL, color="#888888", linestyle="--", linewidth=0.9,
               label="Initial capital")
    ax.set_title(
        f"BTC/USD 4H LightGBM V4 (Risk-Managed)  |  "
        f"size={int(RISK_FRACTION*100)}%  SL={STOP_LOSS_PCT*100:.1f}%  "
        f"TP={TAKE_PROFIT_PCT*100:.1f}%  |  "
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
    n_long_bars = int((p_up > PROB_THRESHOLD).sum())

    print(f"[1/5] Test slice: {test.shape}  "
          f"[{test.index.min()} .. {test.index.max()}]")
    print(f"[2/5] P(up) > {PROB_THRESHOLD:.2f}  ->  {n_long_bars:,} candidate bars  "
          f"(of {len(test):,})")
    print(f"        P(up) range on test: min={p_up.min():.4f}, "
          f"max={p_up.max():.4f}, mean={p_up.mean():.4f}")

    trades = build_trade_log(test, p_up, PROB_THRESHOLD)
    print(f"[3/5] Trade log rows: {len(trades):,}")
    if not trades.empty:
        print(f"        ret_net -> mean={trades['ret_net'].mean():.6f}, "
              f"std={trades['ret_net'].std():.6f}, "
              f"min={trades['ret_net'].min():.6f}, max={trades['ret_net'].max():.6f}")
        print(f"        exit reasons -> "
              f"TP={int((trades['exit_reason']=='TP').sum())}, "
              f"SL={int((trades['exit_reason']=='SL').sum())}, "
              f"TIME={int((trades['exit_reason']=='TIME').sum())}")
        print(f"        avg p_up at entry = {trades['p_up'].mean():.4f}")

    equity = build_equity_curve(test, trades, INITIAL_CAPITAL, RISK_FRACTION)
    metrics = compute_metrics(equity, trades)

    print("\n=== Backtest V4 (4H, Risk-Managed, Out-of-Sample Test) ===")
    print(f"  Threshold         : P(up) > {PROB_THRESHOLD:.2f}")
    print(f"  Position sizing   : {int(RISK_FRACTION*100)}% of equity per trade "
          f"(90% in cash)")
    print(f"  Stop Loss         : {STOP_LOSS_PCT*100:.1f}% below entry")
    print(f"  Take Profit       : {TAKE_PROFIT_PCT*100:.1f}% above entry")
    print(f"  Time exit         : 1 bar (4 hours)")
    print(f"  Round-trip fee    : {FEE_ROUND_TRIP*100:.2f}%")
    print(f"  Initial capital   : ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final equity      : ${metrics['final_equity']:,.2f}")
    print(f"  ROI               : {metrics['roi_pct']:+.2f} %")
    print(f"  Max drawdown      : {metrics['max_drawdown_pct']:.2f} %")
    print(f"  Annualized Sharpe : {metrics['annualized_sharpe']:.2f}")
    print(f"  Total trades      : {metrics['n_trades']:,}")
    print(f"  Win rate          : {metrics['win_rate_pct']:.2f} %")
    print(f"  Avg trade ret     : {metrics['avg_trade_ret_pct']:+.4f} %")
    print(f"  Exit distribution : TP={metrics['n_tp']}  "
          f"SL={metrics['n_sl']}  TIME={metrics['n_time']}")

    plot_equity(equity, metrics)
    print(f"[5/5] Saved equity plot -> {EQUITY_PNG}")


if __name__ == "__main__":
    run()
