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

    for w in [5, 10, 20, 50]:
        sma = close.rolling(w).mean()
        ema = close.ewm(span=w, adjust=False).mean()
        f[f"SMA_{w}_r"] = close / sma
        f[f"EMA_{w}_r"] = close / ema
        f[f"SMA_{w}_slope"] = sma.pct_change(3)

    for p in [1, 2, 3, 5, 10, 20]:
        f[f"ret_{p}d"] = close.pct_change(p)

    for lag in [1, 2, 3, 5, 10]:
        f[f"log_ret_lag{lag}"] = log_ret.shift(lag)

    for w in [5, 10, 20]:
        f[f"vol_{w}"] = log_ret.rolling(w).std()
    f["vol_ratio"] = f["vol_5"] / f["vol_20"]

    for w in [7, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(w).mean()
        loss = (-delta.clip(upper=0)).rolling(w).mean()
        f[f"RSI_{w}"] = 100 - (100 / (1 + gain / loss))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["MACD"] = macd / close
    f["MACD_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    f["BB_pos"] = (close - (sma20 - 2 * std20)) / (4 * std20)
    f["BB_width"] = 4 * std20 / sma20

    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    f["ATR_pct"] = tr.rolling(14).mean() / close

    ll14 = low.rolling(14).min()
    hh14 = high.rolling(14).max()
    f["Stoch_K"] = 100 * (close - ll14) / (hh14 - ll14)
    f["Stoch_D"] = f["Stoch_K"].rolling(3).mean()

    vol_sma20 = volume.rolling(20).mean()
    f["vol_ratio_20"] = volume / vol_sma20
    obv = (np.sign(close.diff()) * volume).cumsum()
    f["OBV_ratio"] = obv / obv.rolling(20).mean()

    f["HL_pct"] = (high - low) / close
    f["OC_pct"] = (open_ - close) / close
    f["Gap"] = (open_ - close.shift()) / close.shift()

    f["above_SMA50"] = (close > close.rolling(50).mean()).astype(int)
    f["golden_cross"] = (
        close.rolling(50).mean() > close.rolling(200).mean()
    ).astype(int)

    f["dow"] = data.index.dayofweek
    f["month"] = data.index.month

    doy = data.index.dayofyear
    f["sin_annual"] = np.sin(2 * np.pi * doy / 252)
    f["cos_annual"] = np.cos(2 * np.pi * doy / 252)

    return f


def monte_carlo_forecast(
    current_price: float,
    first_day_ret: float,
    hist_log_returns: np.ndarray,
    n_days: int = 60,
    n_sims: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """
    Geometric Brownian Motion Monte Carlo.
    Day 1 drift comes from the ML model prediction.
    Subsequent days use historical mean drift + realized volatility.
    Returns array of shape (n_sims, n_days + 1) — column 0 is current_price.
    """
    rng = np.random.default_rng(seed)
    hist_mean = float(hist_log_returns.mean())
    hist_vol = float(hist_log_returns.std())

    paths = np.empty((n_sims, n_days + 1))
    paths[:, 0] = current_price

    for day in range(1, n_days + 1):
        drift = first_day_ret if day == 1 else hist_mean
        shocks = rng.standard_normal(n_sims)
        paths[:, day] = paths[:, day - 1] * np.exp(drift + hist_vol * shocks)

    return paths


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        params = parse_qs(urlparse(self.path).query)
        ticker = params.get("ticker", ["AAPL"])[0].upper()[:10]
        n_days = int(params.get("days", ["60"])[0])
        n_days = max(10, min(n_days, 120))  # clamp 10–120

        try:
            data = yf.download(ticker, period="4y", auto_adjust=True, progress=False)
            if len(data) < 260:
                raise ValueError(f"Not enough historical data for {ticker!r}")

            features = compute_features(data)
            close = data["Close"].squeeze()
            log_ret = np.log(close / close.shift(1))
            target = log_ret.shift(-1)

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

            # Next-day prediction for the point estimate
            last_X = np.nan_to_num(features.iloc[-1:].values.astype(float))
            next_ret = float(model.predict(scaler.transform(last_X))[0])
            current_price = float(close.iloc[-1])
            predicted_price = current_price * np.exp(next_ret)

            # ── Monte Carlo forecast ──────────────────────────────────────────
            hist_log_rets = log_ret.dropna().values
            paths = monte_carlo_forecast(
                current_price=current_price,
                first_day_ret=next_ret,
                hist_log_returns=hist_log_rets,
                n_days=n_days,
                n_sims=1000,
            )
            # paths[:,0] = current_price; paths[:,1..n_days] = simulated prices
            future_prices = paths[:, 1:]  # shape (1000, n_days)

            pctiles = np.percentile(future_prices, [5, 25, 50, 75, 95], axis=0)
            future_dates = pd.bdate_range(
                start=close.index[-1] + pd.Timedelta(days=1), periods=n_days
            )

            # Stacking-friendly data: base + height for each band
            # outer band: p5 → p95; inner band: p25 → p75
            p5, p25, p50, p75, p95 = [pctiles[i] for i in range(5)]
            signal_horizon_days = min(90, n_days)
            signal_idx = signal_horizon_days - 1
            signal_price = float(p50[signal_idx])
            signal_return_pct = ((signal_price - current_price) / current_price) * 100
            signal = "UP" if signal_return_pct > 0 else "DOWN"

            forecast = {
                "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
                "p5":  p5.round(2).tolist(),
                "p25": p25.round(2).tolist(),
                "p50": p50.round(2).tolist(),
                "p75": p75.round(2).tolist(),
                "p95": p95.round(2).tolist(),
                # outer band expressed as base + height for Recharts stacked areas
                "outer_base":   p5.round(2).tolist(),
                "outer_height": (p95 - p5).round(2).tolist(),
                # inner band
                "inner_base":   p25.round(2).tolist(),
                "inner_height": (p75 - p25).round(2).tolist(),
                "n_simulations": 1000,
                "n_days": n_days,
                "signal_horizon_days": signal_horizon_days,
                "signal_price": round(signal_price, 2),
                "signal_return_pct": round(signal_return_pct, 3),
                "signal": signal,
            }

            # Historical context: last 45 days for the forecast chart
            hist_window = close.iloc[-45:]
            history = {
                "dates":  [d.strftime("%Y-%m-%d") for d in hist_window.index],
                "prices": hist_window.round(2).tolist(),
            }

            # ── Original charts (unchanged) ───────────────────────────────────
            n_chart = min(120, len(y_test))
            chart_idx = combined.index[split:][-n_chart:]
            chart_actual = close.reindex(chart_idx).round(2).tolist()
            pred_prices = (
                close_aligned[split:][-n_chart:] * np.exp(test_preds[-n_chart:])
            ).round(2).tolist()

            signal = np.where(test_preds > 0, 1.0, 0.0)
            cum_strat = (np.exp(np.cumsum(signal * y_test)) - 1) * 100
            cum_bh = (np.exp(np.cumsum(y_test)) - 1) * 100
            bt_dates = combined.index[split:].strftime("%Y-%m-%d").tolist()[::3]
            bt_strat = [round(x, 2) for x in cum_strat.tolist()[::3]]
            bt_bh = [round(x, 2) for x in cum_bh.tolist()[::3]]

            result = {
                "ticker": ticker,
                "current_price": round(current_price, 2),
                "predicted_price": round(predicted_price, 2),
                "predicted_return_pct": round(next_ret * 100, 3),
                "direction": "UP" if next_ret > 0 else "DOWN",
                "signal": signal,
                "signal_horizon_days": signal_horizon_days,
                "signal_price": round(signal_price, 2),
                "signal_return_pct": round(signal_return_pct, 3),
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
                "forecast": forecast,
                "history": history,
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

    def log_message(self, fmt, *args):
        pass
