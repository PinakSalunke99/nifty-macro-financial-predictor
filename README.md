# Nifty Macro Financial Predictor

A macro-financial analytics system that predicts the equilibrium value of the Nifty 50 index using a multi-variable regression model derived from macroeconomic research.

## Features

- Real-time market data integration using Yahoo Finance
- Macroeconomic indicators using FRED API
- Regression-based Nifty 50 prediction engine
- Model explainability with factor weightage analysis
- 30-day historical backtesting
- Professional error metrics (MAPE, RMSE)
- Model drift monitoring
- Interactive Streamlit dashboard

## Variables Used

The model integrates the following macro-financial variables:

- USD/INR exchange rate
- Repo Rate
- FII inflows
- S&P 500 Index
- GDP
- Inflation

Regression Equation:

NIFTY =
-10130
+ 2.503 × USDINR
+ 661.7 × Repo Rate
+ 0.001865 × FII
+ 35.25 × SPY
+ 2526 × GDP
+ 67.11 × Inflation

## Installation

```bash
pip install -r requirements.txt