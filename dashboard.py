"""
BTC/USD Quantitative Trading Portfolio - Streamlit Dashboard.

Run with:
    streamlit run dashboard.py

Sidebar sections
----------------
- Project Overview  : narrative + tail sample of the cleaned OHLCV dataset.
- Trading Strategy   : V1 (1m base) -> V3 (4H base) -> V4 (4H risk-managed),
                       with per-version metrics and equity curves, plus a
                       3-way comparison view.
- Feature Importance : 1-minute training reference plot.

All data loads are lazy and memory-aware: we only ever pull the last 100
rows of the cleaned CSV into the browser, and the equity-curve PNGs are
served directly from disk via st.image.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


# ---------- Paths -------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent
CLEAN_PATH: Path = PROJECT_ROOT / "data" / "processed" / "btcusd_cleaned.csv"
FEATURES_PATH: Path = PROJECT_ROOT / "data" / "processed" / "btcusd_features.csv"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FEATURE_IMPORTANCE_PNG: Path = OUTPUTS_DIR / "lgbm_feature_importance.png"

# Three equity-curve PNGs the dashboard reads.
EQUITY_V1_PNG: Path = OUTPUTS_DIR / "portfolio_growth.png"             # 1m base (V1)
EQUITY_4H_PNG: Path = OUTPUTS_DIR / "portfolio_growth_4h.png"         # 4H base (V3)
EQUITY_V4_PNG: Path = OUTPUTS_DIR / "portfolio_growth_v4_risk_managed.png"  # 4H risk-managed (V4)


# ---------- Headline metrics (static snapshots) -------------------------------
# Pulled from the run logs in outputs/. Kept in code so the dashboard renders
# without re-running the pipeline.

# V1: 1m baseline, predict==1 every model-1 bar, full sizing, no SL/TP.
# The geometric compounding annihilated the account in the original run
# (per-trade -0.2% compounded 44k times); final equity is shown as the
# empirically observed value from outputs/_backtest_run.log.
V1_METRICS: dict = {
    "label": "V1 (1m base, full sizing, no SL/TP)",
    "timeframe": "1-minute",
    "initial_capital": 10_000.0,
    "final_equity": 0.00,            # full-compound underflow from outputs/_backtest_run.log
    "roi_pct": -100.00,
    "max_dd_pct": -100.00,
    "annualized_sharpe": -57.58,
    "win_rate_pct": 14.42,
    "total_trades": 43_848,
    "test_period": "2024-04-15 -> 2026-06-16 (1m bars)",
    "signal_threshold": "model predict == 1",
    "position_sizing": "100% of equity per trade",
    "stop_loss": "none",
    "take_profit": "none",
    "fee_model": "0.1% per side (0.2% round-trip)",
    "exit_breakdown": {},
}

# V3: 4H base, full sizing, no SL/TP.
V3_METRICS: dict = {
    "label": "V3 (4H, 100% sizing, no SL/TP)",
    "timeframe": "4-hour",
    "initial_capital": 10_000.0,
    "final_equity": 5_706.25,
    "roi_pct": -42.94,
    "max_dd_pct": -43.86,
    "annualized_sharpe": -1.46,
    "win_rate_pct": 45.90,
    "total_trades": 329,
    "test_period": "2024-04-20 -> 2026-06-16 (4H bars)",
    "signal_threshold": "P(up) > 0.55",
    "position_sizing": "100% of equity per trade",
    "stop_loss": "none",
    "take_profit": "none",
    "fee_model": "0.1% per side (0.2% round-trip)",
    "exit_breakdown": {},
}

# V4: 4H risk-managed.
V4_METRICS: dict = {
    "label": "V4 (4H, 10% sizing, 2% SL / 4% TP, 1-bar time exit)",
    "timeframe": "4-hour",
    "initial_capital": 10_000.0,
    "final_equity": 9_503.68,
    "roi_pct": -4.96,
    "max_dd_pct": -6.01,
    "annualized_sharpe": -1.55,
    "win_rate_pct": 44.98,
    "total_trades": 329,
    "test_period": "2024-04-20 -> 2026-06-16 (4H bars)",
    "signal_threshold": "P(up) > 0.55",
    "position_sizing": "10% of equity per trade (90% in cash)",
    "stop_loss": "2.0% below entry",
    "take_profit": "4.0% above entry",
    "fee_model": "0.1% per side (0.2% round-trip)",
    "exit_breakdown": {"TP": 5, "SL": 41, "TIME": 283},
}


# ---------- Page config -------------------------------------------------------

st.set_page_config(
    page_title="Quantitative Trading Portfolio: BTC/USD AI Model",
    page_icon="\U0001F4C8",  # chart-increasing
    layout="wide",
)


# ---------- Helpers -----------------------------------------------------------

@st.cache_data(show_spinner="Loading cleaned OHLCV (last 100 rows)...")
def load_cleaned_tail(path: Path, n: int = 100) -> pd.DataFrame:
    """Read only the last n rows of the cleaned CSV to keep memory low."""
    return pd.read_csv(path).tail(n)


def file_exists(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def show_image_or_warning(path: Path, caption: str) -> None:
    if file_exists(path):
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Asset not found: {path.name}. Run the relevant pipeline first.")


def fmt_money(x: float) -> str:
    return "n/a" if x != x else f"${x:,.2f}"


def fmt_pct(x: float) -> str:
    return "n/a" if x != x else f"{x:+.2f}%"


def render_metrics_row(m: dict) -> None:
    """Render the 6 headline metric tiles for one strategy variant."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Equity", fmt_money(m["final_equity"]),
              delta=f"{m['roi_pct']:+.2f}% ROI")
    c2.metric("Total Trades", f"{m['total_trades']:,}")
    c3.metric("Win Rate", f"{m['win_rate_pct']:.2f}%")
    c4.metric("Max Drawdown", f"{m['max_dd_pct']:.2f}%")
    c5, c6 = st.columns(2)
    sharpe = m["annualized_sharpe"]
    sharpe_str = f"{sharpe:.2f}" if sharpe == sharpe else "n/a"
    c5.metric("Annualized Sharpe", sharpe_str)
    c6.metric("Initial Capital", f"${m['initial_capital']:,.2f}")


def render_version_block(tag: str, m: dict, png: Path) -> None:
    """Render a full per-version block: header, metrics, equity curve, config."""
    st.subheader(f"{tag} - {m['timeframe']} - {m['label']}")
    render_metrics_row(m)
    show_image_or_warning(png, f"{tag} equity curve")
    with st.expander(f"{tag} configuration", expanded=False):
        st.markdown(
            f"""
            - **Test period**: {m['test_period']}
            - **Signal threshold**: {m['signal_threshold']}
            - **Position sizing**: {m['position_sizing']}
            - **Stop Loss**: {m['stop_loss']}
            - **Take Profit**: {m['take_profit']}
            - **Fee model**: {m['fee_model']}
            - **Hold horizon**: { '1 bar (= 4 hours)' if m['timeframe']=='4-hour' else '1 bar (= 1 minute)' if m['timeframe']=='1-minute' else '-' }
            - **Feature set**: log_return, SMA_50, SMA_200, RSI_14,
              MACD, MACD_signal, MACD_hist, BB_upper, BB_lower
            """
        )
        if m.get("exit_breakdown"):
            eb = m["exit_breakdown"]
            st.markdown(
                f"**Exit-reason mix**: TP={eb.get('TP', 0)}  "
                f"SL={eb.get('SL', 0)}  TIME={eb.get('TIME', 0)}"
            )


# ---------- Sidebar -----------------------------------------------------------

st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to:",
    options=[
        "Project Overview",
        "Trading Strategy Results",
        "Feature Importance",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Project**
    - Asset: BTC/USD
    - Timeframes: 1-minute -> 4-hour
    - Model: LightGBM classifier
    - Target: 1-bar forward direction

    **Pipeline**
    1. `src/preprocess_btcusd.py` - clean & reindex 1m
    2. `src/features_btcusd.py`  - indicators + target
    3. `src/pipeline_4h.py`      - resample, train, V3 backtest
    4. `src/backtest_lgbm_v4_risk.py` - V4 risk-managed backtest
    5. `dashboard.py`            - this view
    """
)


# ---------- Section: Project Overview ----------------------------------------

if section == "Project Overview":
    st.title("Quantitative Trading Portfolio: BTC/USD AI Model")
    st.caption("LightGBM-based mid-frequency trading system on minute-bar Bitcoin data.")

    st.subheader("Executive Summary")
    st.markdown(
        """
        This project builds an end-to-end quantitative trading pipeline for **BTC/USD**:

        - **Data**: 1-minute OHLCV bars spanning 2012-01-01 -> 2026-06-16 (~7.6M rows),
          cleaned to a strict 1-minute grid with forward-fill imputation.
        - **Features**: log-returns, SMA/EMA, RSI, MACD, and Bollinger Bands -
          computed with pure pandas/numpy (no TA-Lib).
        - **Model**: gradient-boosted decision trees (`lightgbm.LGBMClassifier`)
          trained on engineered indicators only, with strict sequential
          train/val/test splits to avoid look-ahead.
        - **Strategy**: long-only gated by a probability threshold with a
          0.1%-per-side fee model.
        - **Risk overlay (V4)**: 10% position sizing with 2% stop-loss and
          4% take-profit, 1-bar time exit, conservative SL-on-give-up
          when both bands are touched in the same bar.

        The dashboard reports the headline metrics from three backtest
        variants (V1, V3, V4) on the out-of-sample test slice.
        """
    )

    st.subheader("Cleaned Dataset (tail)")
    st.caption(
        f"Showing the last 100 rows of `{CLEAN_PATH.relative_to(PROJECT_ROOT)}`. "
        "Used to confirm timestamps are continuous and prices are clean."
    )
    if file_exists(CLEAN_PATH):
        tail = load_cleaned_tail(CLEAN_PATH, n=100)
        st.dataframe(tail, use_container_width=True, height=320)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows shown", f"{len(tail):,}")
        c2.metric("First timestamp", str(tail["Timestamp"].iloc[0]))
        c3.metric("Last timestamp", str(tail["Timestamp"].iloc[-1]))
    else:
        st.error(
            f"Cleaned dataset not found at `{CLEAN_PATH}`. "
            "Run `python src/preprocess_btcusd.py` first."
        )


# ---------- Section: Trading Strategy Results --------------------------------

elif section == "Trading Strategy Results":
    st.title("Trading Strategy Results")
    st.caption("V1 (1m base) -> V3 (4H base) -> V4 (4H risk-managed).")

    # ---------- V1 ----------
    render_version_block("V1", V1_METRICS, EQUITY_V1_PNG)
    st.markdown("---")

    # ---------- V3 ----------
    render_version_block("V3", V3_METRICS, EQUITY_4H_PNG)
    st.markdown("---")

    # ---------- V4 ----------
    render_version_block("V4", V4_METRICS, EQUITY_V4_PNG)

    # ---------- 3-way comparison ----------
    st.markdown("---")
    st.subheader("V1 vs V3 vs V4 - Side-by-Side")
    cmp = pd.DataFrame(
        {
            "Metric": [
                "Final Equity ($)",
                "ROI (%)",
                "Max Drawdown (%)",
                "Annualized Sharpe",
                "Win Rate (%)",
                "Total Trades",
                "Position Sizing",
                "Stop Loss",
                "Take Profit",
            ],
            "V1 (1m base)": [
                fmt_money(V1_METRICS["final_equity"]),
                fmt_pct(V1_METRICS["roi_pct"]),
                fmt_pct(V1_METRICS["max_dd_pct"]),
                f"{V1_METRICS['annualized_sharpe']:.2f}" if V1_METRICS["annualized_sharpe"] == V1_METRICS["annualized_sharpe"] else "n/a",
                f"{V1_METRICS['win_rate_pct']:.2f}",
                f"{V1_METRICS['total_trades']:,}",
                V1_METRICS["position_sizing"],
                V1_METRICS["stop_loss"],
                V1_METRICS["take_profit"],
            ],
            "V3 (4H base)": [
                fmt_money(V3_METRICS["final_equity"]),
                fmt_pct(V3_METRICS["roi_pct"]),
                fmt_pct(V3_METRICS["max_dd_pct"]),
                f"{V3_METRICS['annualized_sharpe']:.2f}" if V3_METRICS["annualized_sharpe"] == V3_METRICS["annualized_sharpe"] else "n/a",
                f"{V3_METRICS['win_rate_pct']:.2f}",
                f"{V3_METRICS['total_trades']:,}",
                V3_METRICS["position_sizing"],
                V3_METRICS["stop_loss"],
                V3_METRICS["take_profit"],
            ],
            "V4 (4H risk)": [
                fmt_money(V4_METRICS["final_equity"]),
                fmt_pct(V4_METRICS["roi_pct"]),
                fmt_pct(V4_METRICS["max_dd_pct"]),
                f"{V4_METRICS['annualized_sharpe']:.2f}" if V4_METRICS["annualized_sharpe"] == V4_METRICS["annualized_sharpe"] else "n/a",
                f"{V4_METRICS['win_rate_pct']:.2f}",
                f"{V4_METRICS['total_trades']:,}",
                V4_METRICS["position_sizing"],
                V4_METRICS["stop_loss"],
                V4_METRICS["take_profit"],
            ],
        }
    )
    st.dataframe(cmp, use_container_width=True, hide_index=True)

    # Headline impact numbers
    c1, c2, c3 = st.columns(3)
    v1_to_v4_dd = V1_METRICS["max_dd_pct"] - V4_METRICS["max_dd_pct"]
    v3_to_v4_dd = V3_METRICS["max_dd_pct"] - V4_METRICS["max_dd_pct"]
    v3_to_v4_roi = V4_METRICS["roi_pct"] - V3_METRICS["roi_pct"]
    c1.metric("V1 -> V4 drawdown reduction", f"{v1_to_v4_dd:+.2f} pp")
    c2.metric("V3 -> V4 drawdown reduction", f"{v3_to_v4_dd:+.2f} pp")
    c3.metric("V3 -> V4 ROI improvement", f"{v3_to_v4_roi:+.2f} pp")


# ---------- Section: Feature Importance --------------------------------------

elif section == "Feature Importance":
    st.title("Feature Importance")
    st.caption("Reference plot from the 1-minute LightGBM training run.")
    show_image_or_warning(
        FEATURE_IMPORTANCE_PNG,
        "1-minute LightGBM feature importance (gain)",
    )
    st.caption(
        "The 4H pipeline writes its own model; this 1m plot is the canonical "
        "feature-importance reference for the project. Run `src/train_lgbm.py` "
        "or `src/pipeline_4h.py` to refresh the underlying artefacts."
    )


# ---------- Footer ------------------------------------------------------------

st.markdown("---")
st.caption(
    "Built with Streamlit. Data and model artefacts are produced by the "
    "`src/` pipeline scripts; this dashboard is a read-only view."
)
