import json
import warnings
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


def compute_features(data: pd.DataFrame) -> pd.DataFrame:
    close = data["Close"].squeeze()
    high = data["High"].squeeze()
    low = data["Low"].squeeze()
    volume = data["Volume"].squeeze()
    open_ = data["Open"].squeeze()
    log_ret = np.log(close / close.shift(1))

    f = pd.DataFrame(index=data.index)

    # MA ratios + slopes
    for w in [5, 10, 20, 50]:
        sma = close.rolling(w).mean()
        ema = close.ewm(span=w, adjust=False).mean()
        f[f"SMA_{w}_r"] = close / sma
        f[f"EMA_{w}_r"] = close / ema
        f[f"SMA_{w}_slope"] = sma.pct_change(3)

    # Momentum
    for p in [1, 2, 3, 5, 10, 20]:
        f[f"ret_{p}d"] = close.pct_change(p)

    # Lagged log returns
    for lag in [1, 2, 3, 5, 10]:
        f[f"log_ret_lag{lag}"] = log_ret.shift(lag)

    # Realized volatility
    for w in [5, 10, 20]:
        f[f"vol_{w}"] = log_ret.rolling(w).std()
    f["vol_ratio"] = f["vol_5"] / f["vol_20"]

    # RSI
    for w in [7, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(w).mean()
        loss = (-delta.clip(upper=0)).rolling(w).mean()
        f[f"RSI_{w}"] = 100 - (100 / (1 + gain / loss))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["MACD"] = macd / close
    f["MACD_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    f["BB_pos"] = (close - (sma20 - 2 * std20)) / (4 * std20)
    f["BB_width"] = 4 * std20 / sma20

    # ATR (normalized)
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    f["ATR_pct"] = tr.rolling(14).mean() / close

    # Stochastic
    ll14 = low.rolling(14).min()
    hh14 = high.rolling(14).max()
    f["Stoch_K"] = 100 * (close - ll14) / (hh14 - ll14)
    f["Stoch_D"] = f["Stoch_K"].rolling(3).mean()

    # Volume
    vol_sma20 = volume.rolling(20).mean()
    f["vol_ratio_20"] = volume / vol_sma20
    obv = (np.sign(close.diff()) * volume).cumsum()
    f["OBV_ratio"] = obv / obv.rolling(20).mean()

    # Price structure
    f["HL_pct"] = (high - low) / close
    f["OC_pct"] = (open_ - close) / close
    f["Gap"] = (open_ - close.shift()) / close.shift()

    # Regime
    f["above_SMA50"] = (close > close.rolling(50).mean()).astype(int)
    f["golden_cross"] = (
        close.rolling(50).mean() > close.rolling(200).mean()
    ).astype(int)

    # Calendar
    f["dow"] = data.index.dayofweek
    f["month"] = data.index.month

    # Fourier seasonality
    doy = data.index.dayofyear
    f["sin_annual"] = np.sin(2 * np.pi * doy / 252)
    f["cos_annual"] = np.cos(2 * np.pi * doy / 252)

    return f


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        params = parse_qs(urlparse(self.path).query)
        ticker = params.get("ticker", ["AAPL"])[0].upper()[:10]

        try:
            data = yf.download(ticker, period="4y", auto_adjust=True, progress=False)
            if len(data) < 260:
                raise ValueError(f"Not enough historical data for {ticker!r}")

            features = compute_features(data)
            close = data["Close"].squeeze()
            log_ret = np.log(close / close.shift(1))
            target = log_ret.shift(-1)  # next-day log return

            combined = features.copy()
            combined["__y__"] = target
            combined = combined.dropna()

            X = combined.drop("__y__", axis=1).values.astype(float)
            y = combined["__y__"].values.astype(float)
            close_aligned = close.reindex(combined.index).values

            scaler = StandardScaler()
            X_sc = scaler.fit_transform(X)

            split = int(len(X_sc) * 0.8)

            model = GradientBoostingRegressor(
                n_estimators=150,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=10,
                random_state=42,
            )
            model.fit(X_sc[:split], y[:split])

            # Test-set metrics
            test_preds = model.predict(X_sc[split:])
            y_test = y[split:]
            dir_acc = float(np.mean(np.sign(y_test) == np.sign(test_preds)))

            # Next-day prediction (last available feature row)
            last_X = np.nan_to_num(features.iloc[-1:].values.astype(float))
            next_ret = float(model.predict(scaler.transform(last_X))[0])
            current_price = float(close.iloc[-1])
            predicted_price = current_price * np.exp(next_ret)

            # Chart data: last 120 days of test set
            n_chart = min(120, len(y_test))
            chart_idx = combined.index[split:][-n_chart:]
            chart_actual = close.reindex(chart_idx).round(2).tolist()
            pred_prices = (
                close_aligned[split:][-n_chart:] * np.exp(test_preds[-n_chart:])
            ).round(2).tolist()

            # Backtest: long when model predicts up, flat otherwise
            signal = np.where(test_preds > 0, 1.0, 0.0)
            cum_strat = (np.exp(np.cumsum(signal * y_test)) - 1) * 100
            cum_bh = (np.exp(np.cumsum(y_test)) - 1) * 100

            # Downsample backtest to every 3rd point to keep payload small
            bt_dates = combined.index[split:].strftime("%Y-%m-%d").tolist()[::3]
            bt_strat = [round(x, 2) for x in cum_strat.tolist()[::3]]
            bt_bh = [round(x, 2) for x in cum_bh.tolist()[::3]]

            result = {
                "ticker": ticker,
                "current_price": round(current_price, 2),
                "predicted_price": round(predicted_price, 2),
                "predicted_return_pct": round(next_ret * 100, 3),
                "direction": "UP" if next_ret > 0 else "DOWN",
                "directional_accuracy": round(dir_acc, 4),
                "chart": {
                    "dates": [d.strftime("%Y-%m-%d") for d in chart_idx],
                    "actual": chart_actual,
                    "predicted": pred_prices,
                },
                "backtest": {
                    "dates": bt_dates,
                    "strategy": bt_strat,
                    "buy_hold": bt_bh,
                },
            }

            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        except Exception as exc:
            body = json.dumps({"error": str(exc)}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, fmt, *args):  # silence access logs
        pass
