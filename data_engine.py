import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

class NiftyPredictorEngine:
    def __init__(self, fred_api_key=None):
        # Precise coefficients from Pinak's Research
        self.intercept = -10130.0
        self.coeffs = {
            "USDINR": 2.503, "Repo": 661.7, "FII": 0.001865,
            "SPY": 35.25, "GDP": 2526.0, "Inflation": 67.11
        }
        self.fred = Fred(api_key=fred_api_key) if fred_api_key else None

    def fetch_data(self):
        """Fetches market data with fallback for fear index analysis."""
        try:
            tickers = ["^NSEI", "SPY", "INR=X", "^INDIAVIX"]
            data = yf.download(tickers, period="45d", interval="1d")['Close'].dropna()
            
            latest = {
                "Nifty_Actual": data['^NSEI'].iloc[-1],
                "SPY_Price": data['SPY'].iloc[-1],
                "USDINR": data['INR=X'].iloc[-1],
                "VIX": data['^INDIAVIX'].iloc[-1],
                "History": data
            }
            return latest, "Success"
        except Exception as e:
            return None, str(e)

    def fetch_live_macro(self):
        """Fetches India-specific macro data from FRED."""
        # Baseline fallbacks from your preferred code
        macro = {"Repo_Rate": 6.50, "FII_Inflow": 1250.0, "GDP_Trillion": 3.9, "Inflation": 4.8}
        
        if not self.fred:
            return macro, "Using Local Baseline (No API Key)"

        try:
            # India CPI (Inflation Proxy) - Series: INDCPIALLMINMEI
            inf_series = self.fred.get_series('INDCPIALLMINMEI')
            # India GDP (Proxy/Annual Growth) - Series: MKTGDPINA646NWDB
            gdp_series = self.fred.get_series('MKTGDPINA646NWDB')
            
            # Update macro dictionary with latest available FRED data
            macro["Inflation"] = round(inf_series.pct_change(periods=12).iloc[-1] * 100, 2)
            macro["GDP_Trillion"] = round(gdp_series.iloc[-1] / 1e12, 2) 
            
            return macro, "Live Macro Data Sync Success"
        except Exception as e:
            return macro, f"Macro API Fallback Active: {str(e)}"

    def predict(self, m_data, s_data):
        """Calculates equilibrium price and normalized weightage."""
        impacts = {
            "USD/INR Impact": self.coeffs["USDINR"] * m_data["USDINR"],
            "US Market Impact": self.coeffs["SPY"] * m_data["SPY_Price"],
            "Monetary Policy": self.coeffs["Repo"] * s_data["Repo_Rate"],
            "Economic Growth": self.coeffs["GDP"] * s_data["GDP_Trillion"],
            "Inflation Effect": self.coeffs["Inflation"] * s_data["Inflation"],
            "Foreign Investment": self.coeffs["FII"] * s_data["FII_Inflow"]
        }
        total_pred = self.intercept + sum(impacts.values())
        
        # Normalized Weightage Improvement
        total_abs_impact = sum(abs(v) for v in impacts.values())
        norm_weights = {k: (abs(v) / total_abs_impact) * 100 for k, v in impacts.items()}
        max_var = max(norm_weights, key=norm_weights.get)
        
        return round(total_pred, 2), impacts, norm_weights, (max_var, round(norm_weights[max_var], 2))

    def get_backtest_with_drift(self, hist_data, s_data):
        """Generates 30 days of predictions and checks for statistical drift."""
        results = []
        errors = []
        for date, row in hist_data.tail(30).iterrows():
            m_tmp = {"SPY_Price": row["SPY"], "USDINR": row["INR=X"]}
            pred, _, _, _ = self.predict(m_tmp, s_data)
            actual = row["^NSEI"]
            results.append({"Date": date, "Actual": actual, "Predicted": pred})
            errors.append(abs(actual - pred) / actual)
            
        df = pd.DataFrame(results)
        mape = mean_absolute_percentage_error(df['Actual'], df['Predicted']) * 100
        rmse = np.sqrt(mean_squared_error(df['Actual'], df['Predicted']))
        
        recent_err = np.mean(errors[-5:])
        baseline_err = np.mean(errors[:-5])
        drift = "High" if (recent_err > baseline_err * 1.5) else "Stable"
        
        return df, round(mape, 2), round(rmse, 2), drift, round(recent_err * 100, 2)