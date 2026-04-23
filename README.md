# AAPL Penalized Regression — Out-of-Sample Stock Price Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white" />
</p>

<p align="center">
  <strong>Predicting Apple (AAPL) closing prices using penalized regression.</strong><br>
  Trained on 2015–2018, forward-predicted through all of 2019 with recursive one-step-ahead simulation.<br>
  Rigorous evaluation: time-series CV, RMSE/MAE/MAPE/R², monthly error analysis.
</p>

---

## Overview

This project applies **penalized linear regression** (ElasticNet) to predict AAPL daily closing prices from OHLCV data. Rather than fitting a black-box model to all available data, it uses a proper **out-of-sample design**: the model trains exclusively on 2015–2018 and then attempts to forecast every trading day in 2019 — without ever peeking at 2019 prices during training.

### Why Penalized Regression?

| Concern | Solution |
|---------|----------|
| Overfitting with noisy financial data | L1/L2 regularization prevents coefficients from exploding |
| Feature selection | ElasticNet (L1 penalty) can drive irrelevant features to exactly zero |
| Interpretable model | Linear coefficients show relative importance of OHLCV inputs |
| No training on future data | TimeSeriesSplit cross-validation ensures temporal integrity |

## Methodology

### Data Pipeline

```
Historical OHLCV Data (2015–2019)
         │
         ▼
    Per-day Features
  ┌──────────────────┐
  │ Open Price       │
  │ High Price       │
  │ Low Price        │
  │ Close Price      │
  │ Volume           │
  └──────────────────┘
         │
         ▼
    Standardization (StandardScaler)
    Fit on training set ONLY
         │
         ▼
    Target: Next-day Close Price
    (shifted by -1 to ensure no leakage)
```

### Training Protocol

```
Training Set: Jan 2015 — Dec 2018 (all trading days)
Test Set:     Jan 2019 — Dec 2019 (held out, never seen during training)

Cross-Validation: TimeSeriesSplit (5 folds)
  Fold 1: Train [Jan '15 – Jun '15] → Test [Jul '15 – Dec '15]
  Fold 2: Train [Jan '15 – Dec '15] → Test [Jan '16 – Jun '16]
  Fold 3: Train [Jan '15 – Jun '16] → Test [Jul '16 – Dec '16]
  ...and so on (no future data leakage)

Hyperparameter Search: GridSearchCV
  α (regularization strength): [0.01, 0.1, 1.0, 10.0]
  l1_ratio (L1 vs L2 balance):  [0.1, 0.5, 0.7, 0.9, 0.95]
  → 20 total combinations evaluated
```

### Forward Prediction Simulation

After finding the best model via cross-validation on training data, the project runs a **recursive one-step-ahead simulation**:

1. Start from last trading day of 2018 (real OHLCV + close)
2. For each day in 2019:
   - Use real open/high/low/volume for that day
   - Feed into trained model → get **predicted next-day close**
   - Update state: use the *predicted* close as input for the next day's prediction
3. Compare all predicted closes against actual closes

This simulates a live system where you know today's opening price but must predict the closing price — and crucially, your prediction feeds into tomorrow's prediction (error accumulation).

### Evaluation Metrics

| Metric | Formula | What It Tells You |
|--------|---------|-------------------|
| **RMSE** | √(Σ(actual - pred)² / n) | Average magnitude of error in dollars |
| **MAE** | Σ\|actual - pred\| / n | Mean absolute dollar error (less sensitive to outliers than RMSE) |
| **MAPE** | mean(\|error\| / \|actual\|) × 100 | Average percentage error — interpretable across price levels |
| **R²** | 1 - SS_res / SS_tot | Fraction of variance explained (1.0 = perfect, 0.0 = baseline mean) |

### Monthly Breakdown

Each month's MAPE, RMSE, and return comparison are computed separately, revealing whether the model performs consistently or degrades in specific market conditions (e.g., Q1 2019 rally vs. Q3 2019 volatility).

## Project Structure

```
AAPL-Penalized-Regression-Model/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/
│   └── aapl_ohlcv.csv                # Apple OHLCV data (2015–2019)
├── src/
│   ├── __init__.py
│   ├── data.py                       # Data loading, feature engineering
│   ├── model.py                      # ElasticNet training + grid search
│   ├── predictor.py                  # Recursive forward simulation
│   └── metrics.py                    # RMSE, MAE, MAPE, R², monthly analysis
├── plots/
│   ├── __init__.py
│   ├── price_comparison.py           # Actual vs predicted prices (line chart)
│   ├── returns_comparison.py         # Cumulative returns (%) 
│   ├── quarterly_analysis.py         # Quarterly MAPE + returns bars
│   └── error_heatmap.py              # Error pattern heatmap by month/day
├── tests/
│   ├── test_data.py
│   ├── test_metrics.py
│   └── test_predictor.py
├── run_prediction.py                  # Entry point — runs full pipeline
└── output/                            # Generated plots and reports
    ├── aapl_2019_prediction.png
    ├── aapl_2019_returns.png
    ├── aapl_2019_quarterly_analysis.png
    └── aapl_2019_error_heatmap.png
```

## Quick Start

### Installation

```bash
git clone https://github.com/shaunakseth5/AAPL-Penalized-Regression-Model.git
cd AAPL-Penalized-Regression-Model
pip install -r requirements.txt
```

### Run the Prediction Pipeline

```bash
python run_prediction.py
```

This will:
1. Load AAPL OHLCV data (2015–2019)
2. Train ElasticNet with TimeSeriesSplit cross-validation
3. Forward-predict all of 2019 recursively
4. Calculate RMSE, MAE, MAPE, R²
5. Generate monthly/quarterly breakdowns
6. Save 4 visualization plots to `output/`

### Custom Configuration

```bash
python run_prediction.py \
    --data-path ./data/aapl_ohlcv.csv \
    --train-end "2018-12-31" \
    --test-start "2019-01-01" \
    --test-end "2019-12-31" \
    --alpha-grid 0.01,0.1,1.0,10.0 \
    --l1-ratio-grid 0.1,0.5,0.7,0.9,0.95 \
    --n-splits 5
```

## Getting AAPL Data

Download historical AAPL OHLCV data from Yahoo Finance:

```python
import yfinance as yf

# Download 2015-2019 daily data
df = yf.download("AAPL", start="2015-01-01", end="2019-12-31")
df.to_csv("data/aapl_ohlcv.csv")
```

Required CSV format: columns `Date,Open,High,Low,Close,Volume` with a datetime index.

## Key Results (from 2019 forward prediction)

> Run `run_prediction.py` to generate your own results.

The model produces daily closing price predictions for all trading days in 2019. The forward-prediction simulation captures the compounding nature of sequential prediction — each day's predicted close feeds into the next day's features, making this a realistic assessment of how the model would perform in a live setting.

## Design Decisions

### Why ElasticNet over Lasso or Ridge?
- **Lasso (L1 only)** can drive coefficients to zero but often struggles with correlated features (OHLC prices are highly collinear)
- **Ridge (L2 only)** keeps all features but doesn't perform feature selection
- **ElasticNet** combines both: it can eliminate irrelevant features (like Lasso) while handling multicollinearity gracefully (like Ridge). The `l1_ratio` parameter controls this trade-off, and the grid search finds the optimal balance.

### Why TimeSeriesSplit Instead of Random CV?
Random k-fold splits would allow future data to leak into training folds (e.g., testing on January while training includes December). TimeSeriesSplit enforces chronological ordering: each fold's test set comes *after* its training set, matching real-world forecasting conditions.

### Why Recursive Forward Prediction Instead of Parallel Prediction?
Parallel prediction uses actual 2019 OHLCV to predict all 2019 closes independently — which is impossible in live trading since you don't have tomorrow's close today. The recursive simulation mimics a real system: you observe today's open/high/low, predict the close, then use that prediction (rather than the actual close) as input for tomorrow's prediction.

### Why Standardize Features?
ElasticNet applies the same penalty strength across all features. Without standardization, a feature with large values (e.g., volume in millions) would be penalized disproportionately compared to a small-scale feature (e.g., price spread). StandardScaler ensures each feature contributes equally to the regularization term.

## Limitations

- **Single stock:** Only tested on AAPL — results may not generalize to other equities
- **5 features only:** OHLCV alone; no macroeconomic indicators, sector data, or sentiment
- **Linear model:** Cannot capture non-linear patterns (e.g., volatility clustering, regime shifts)
- **2019 only:** Specific market conditions of that year (post-election rally, low rates)
- **No transaction costs:** Predictions are theoretical; slippage and bid-ask spread not modeled

## License

MIT — use it as a template for your own predictive models.

---

Built and maintained by [Shaunak Seth](https://github.com/shaunakseth5)  
*Systematic prediction from math to market.*
