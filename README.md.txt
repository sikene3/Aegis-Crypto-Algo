# 📈 Quantitative Trading System: BTC/USD AI Model

## 🎯 Executive Summary
This project demonstrates a complete end-to-end algorithmic trading pipeline. It transitions from analyzing high-frequency 1-minute order book data to developing a stable, risk-managed 4-Hour swing trading strategy using **LightGBM** and strict financial risk management principles.

## 🛠️ Tech Stack & Methodology
* **Data Processing:** Pandas, NumPy (Resampled 7.6+ Million rows of 1-minute data to 4H).
* **Feature Engineering:** Vectorized calculation of RSI, MACD, Bollinger Bands, and Moving Averages.
* **Machine Learning:** `LightGBM` Classifier with Time-Series Cross-Validation to completely prevent Data Leakage.
* **Backtesting Engine:** Custom-built Python engine simulating real-world exchange fees (0.1%), position sizing, and slippage.
* **Visualization:** Matplotlib, Streamlit (Interactive Dashboard).

## 📉 The Journey: From Failure to Alpha
### Version 1 & 2: The High-Frequency Trap
Initially, the model was trained to predict 15-minute price movements. While it achieved a statistical edge, the backtest revealed a critical financial flaw: **Fee Erosion**. Executing ~44,000 trades meant that the 0.1% exchange fee completely consumed the portfolio.

### Version 3: The 4-Hour Shift & The Gambler's Ruin
The data was resampled to a 4-Hour timeframe to capture larger moves. The model's win rate spiked to **45.9%**. However, executing trades with 100% position sizing led to severe drawdowns during losing streaks.

### Version 4: Production-Ready Risk Management
To create a mathematically sound portfolio, strict risk constraints were applied to the AI signals:
1. **Probability Filter:** The model only triggers a LONG signal if the prediction confidence is strictly `> 0.55`.
2. **Position Sizing:** Only **10% of total equity** is risked per trade.
3. **Stop-Loss (SL):** Hard exit at `-2.0%` to protect capital.
4. **Take-Profit (TP):** Hard exit at `+4.0%` for a 1:2 Risk/Reward ratio.

## 🚀 How to Run the Project
1. Install requirements: `pip install -r requirements.txt`
2. Run the data pipeline: `python src/data_pipeline.py`
3. Launch the interactive dashboard: `streamlit run dashboard.py`

*Disclaimer: This project is for educational and portfolio purposes only. It does not constitute financial advice.*