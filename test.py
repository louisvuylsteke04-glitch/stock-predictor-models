import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print("XGBoost unavailable (brew install libomp to fix)")

try:
    import lightgbm as lgb

    HAS_LGB = True
except Exception:
    HAS_LGB = False
    print("LightGBM not installed (pip install lightgbm)")


# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_rsi(series, window):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - (100 / (1 + gain / loss))


def max_drawdown(cum_ret_series):
    equity = 1 + cum_ret_series
    rolling_max = np.maximum.accumulate(equity)
    return ((equity - rolling_max) / rolling_max).min()


# ── Feature engineering ───────────────────────────────────────────────────────
def add_features(df, market_data=None):
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    volume = df["Volume"].squeeze()
    open_ = df["Open"].squeeze()
    log_ret = np.log(close / close.shift(1))

    # Moving averages + slopes + ratios
    for w in [5, 10, 20, 50, 100, 200]:
        df[f"SMA_{w}"] = close.rolling(w).mean()
        df[f"EMA_{w}"] = close.ewm(span=w, adjust=False).mean()
        df[f"Price_SMA_{w}_ratio"] = close / df[f"SMA_{w}"]
        df[f"Price_EMA_{w}_ratio"] = close / df[f"EMA_{w}"]
        df[f"SMA_{w}_slope"] = df[f"SMA_{w}"].pct_change(5)

    # Momentum / rate of change
    for p in [1, 2, 3, 5, 10, 15, 20, 60]:
        df[f"Return_{p}d"] = close.pct_change(p)

    # Lagged price, returns, volume
    for lag in [1, 2, 3, 4, 5, 10, 21]:
        df[f"Close_lag_{lag}"] = close.shift(lag)
        df[f"Return_lag_{lag}"] = log_ret.shift(lag)
        df[f"Volume_lag_{lag}"] = volume.shift(lag)

    # Realized volatility
    for w in [5, 10, 20, 60]:
        df[f"RealVol_{w}"] = log_ret.rolling(w).std() * np.sqrt(252)
    df["Vol_ratio_5_20"] = df["RealVol_5"] / df["RealVol_20"]

    # RSI
    for w in [7, 14, 28]:
        df[f"RSI_{w}"] = compute_rsi(close, w)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    df["MACD_cross"] = (df["MACD"] > df["MACD_Signal"]).astype(int)

    # Bollinger Bands + squeeze signal
    for w in [10, 20]:
        sma = close.rolling(w).mean()
        std = close.rolling(w).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        df[f"BB_width_{w}"] = (upper - lower) / sma
        df[f"BB_pos_{w}"] = (close - lower) / (upper - lower)
        df[f"BB_squeeze_{w}"] = (
            df[f"BB_width_{w}"] < df[f"BB_width_{w}"].rolling(50).quantile(0.2)
        ).astype(int)

    # ATR (normalized)
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()
    df["ATR_pct"] = df["ATR_14"] / close

    # Stochastic
    for w in [9, 14, 21]:
        ll = low.rolling(w).min()
        hh = high.rolling(w).max()
        df[f"Stoch_K_{w}"] = 100 * (close - ll) / (hh - ll)
        df[f"Stoch_D_{w}"] = df[f"Stoch_K_{w}"].rolling(3).mean()

    # CCI
    typical = (high + low + close) / 3
    for w in [14, 20]:
        mean_dev = typical.rolling(w).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df[f"CCI_{w}"] = (typical - typical.rolling(w).mean()) / (0.015 * mean_dev)

    # Williams %R
    hh14 = high.rolling(14).max()
    ll14 = low.rolling(14).min()
    df["Williams_R"] = -100 * (hh14 - close) / (hh14 - ll14)

    # Volume indicators
    df["Volume_SMA_20"] = volume.rolling(20).mean()
    df["Volume_ratio"] = volume / df["Volume_SMA_20"]
    df["Volume_trend_5d"] = volume.pct_change(5)
    df["OBV"] = (np.sign(close.diff()) * volume).cumsum()
    df["OBV_EMA_20"] = df["OBV"].ewm(span=20, adjust=False).mean()
    df["OBV_ratio"] = df["OBV"] / df["OBV_EMA_20"]
    df["VWAP_20"] = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
    df["VWAP_ratio"] = close / df["VWAP_20"]

    # Price structure
    df["HL_pct"] = (high - low) / close
    df["OC_pct"] = (open_ - close) / close
    df["Gap"] = (open_ - close.shift()) / close.shift()

    # Donchian channels
    for w in [20, 55]:
        dc_hi = high.rolling(w).max()
        dc_lo = low.rolling(w).min()
        df[f"DC_pos_{w}"] = (close - dc_lo) / (dc_hi - dc_lo)

    # Market regime
    df["Above_SMA_50"] = (close > df["SMA_50"]).astype(int)
    df["Above_SMA_200"] = (close > df["SMA_200"]).astype(int)
    df["Golden_cross"] = (df["SMA_50"] > df["SMA_200"]).astype(int)
    df["Trend_strength"] = (close - df["SMA_20"]).abs() / (df["RealVol_20"] * close / np.sqrt(252))

    # 52-week range position
    hh52 = high.rolling(252).max()
    ll52 = low.rolling(252).min()
    df["Price_52w_pos"] = (close - ll52) / (hh52 - ll52)

    # Fourier seasonality
    doy = df.index.dayofyear
    df["Sin_annual"] = np.sin(2 * np.pi * doy / 252)
    df["Cos_annual"] = np.cos(2 * np.pi * doy / 252)
    df["Sin_semiannual"] = np.sin(4 * np.pi * doy / 252)
    df["Cos_semiannual"] = np.cos(4 * np.pi * doy / 252)

    # Calendar
    df["DayOfWeek"] = df.index.dayofweek
    df["Month"] = df.index.month
    df["Quarter"] = df.index.quarter
    df["DayOfMonth"] = df.index.day

    # Cross-asset features
    if market_data is not None:
        for name, mdf in market_data.items():
            mc = mdf["Close"].squeeze().reindex(df.index, method="ffill")
            df[f"{name}_ret_1d"] = mc.pct_change()
            df[f"{name}_ret_5d"] = mc.pct_change(5)
            df[f"{name}_SMA20_ratio"] = mc / mc.rolling(20).mean()
            df[f"Corr_{name}_20"] = (
                close.pct_change().rolling(20).corr(mc.pct_change())
            )

    return df


# ── Download data ─────────────────────────────────────────────────────────────
TICKER = "AAPL"
print(f"Downloading {TICKER} + market context data...")
data = yf.download(TICKER, start="2010-01-01", end="2024-12-31", auto_adjust=True)
print(f"  {TICKER}: {len(data)} trading days")

market_data = {}
for t in ["SPY", "QQQ", "^VIX"]:
    mdf = yf.download(t, start="2010-01-01", end="2024-12-31", auto_adjust=True)
    label = t.replace("^", "")
    market_data[label] = mdf
    print(f"  {label}: {len(mdf)} days")

# ── Features & target ─────────────────────────────────────────────────────────
data = add_features(data, market_data)

close_series = data["Close"].squeeze()
data["Log_Return"] = np.log(close_series / close_series.shift(1))
data["Target_return"] = data["Log_Return"].shift(-1)   # next day log return
data["Target_price"] = close_series.shift(-1)          # for $ evaluation

data = data.dropna()

EXCLUDE = {
    "Target_return", "Target_price", "Log_Return",
    "Close", "Open", "High", "Low", "Volume",
}
feature_cols = [c for c in data.columns if c not in EXCLUDE]
print(f"\n{len(feature_cols)} features generated")

X = data[feature_cols].values.astype(float)
y_ret = data["Target_return"].values.astype(float)
y_price = data["Target_price"].values.astype(float)
close_vals = close_series.reindex(data.index).values
dates = data.index

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Model definitions ─────────────────────────────────────────────────────────
base_models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=400, max_depth=10, min_samples_leaf=5,
        max_features="sqrt", random_state=42, n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.8, min_samples_leaf=10, random_state=42,
    ),
}
if HAS_XGB:
    base_models["XGBoost"] = XGBRegressor(
        n_estimators=600, max_depth=5, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0,
    )
if HAS_LGB:
    base_models["LightGBM"] = lgb.LGBMRegressor(
        n_estimators=600, max_depth=5, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=-1,
    )

# ── Walk-forward cross-validation ─────────────────────────────────────────────
print("\nWalk-forward validation (8 expanding windows)...")
tscv = TimeSeriesSplit(n_splits=8)
cv_results = {}

for name, model in base_models.items():
    maes, rmses, dirs = [], [], []
    for train_idx, test_idx in tscv.split(X_scaled):
        model.fit(X_scaled[train_idx], y_ret[train_idx])
        pred_ret = model.predict(X_scaled[test_idx])
        actual_ret = y_ret[test_idx]
        pred_price = close_vals[test_idx] * np.exp(pred_ret)
        actual_price = y_price[test_idx]
        maes.append(mean_absolute_error(actual_price, pred_price))
        rmses.append(np.sqrt(mean_squared_error(actual_price, pred_price)))
        dirs.append(np.mean(np.sign(actual_ret) == np.sign(pred_ret)))
    cv_results[name] = dict(MAE=np.mean(maes), RMSE=np.mean(rmses), DirAcc=np.mean(dirs))
    print(
        f"  {name:22s}  MAE=${cv_results[name]['MAE']:.2f}  "
        f"RMSE=${cv_results[name]['RMSE']:.2f}  "
        f"DirAcc={cv_results[name]['DirAcc']:.1%}"
    )

# ── Stacking ensemble (Ridge meta-learner over OOF predictions) ───────────────
print("\nBuilding stacking ensemble...")
split = int(len(X_scaled) * 0.8)
inner_cv = TimeSeriesSplit(n_splits=5)

oof_preds = np.zeros((split, len(base_models)))
for i, (name, model) in enumerate(base_models.items()):
    oof = np.zeros(split)
    for tr_idx, val_idx in inner_cv.split(X_scaled[:split]):
        model.fit(X_scaled[:split][tr_idx], y_ret[:split][tr_idx])
        oof[val_idx] = model.predict(X_scaled[:split][val_idx])
    oof_preds[:, i] = oof

test_preds = np.zeros((len(X_scaled) - split, len(base_models)))
for i, (name, model) in enumerate(base_models.items()):
    model.fit(X_scaled[:split], y_ret[:split])
    test_preds[:, i] = model.predict(X_scaled[split:])

meta = Ridge(alpha=1.0)
meta.fit(oof_preds, y_ret[:split])
stacked_pred_ret = meta.predict(test_preds)

y_test = y_price[split:]
y_test_ret = y_ret[split:]
close_test = close_vals[split:]
dates_test = dates[split:]

stacked_pred_price = close_test * np.exp(stacked_pred_ret)

mae = mean_absolute_error(y_test, stacked_pred_price)
rmse = np.sqrt(mean_squared_error(y_test, stacked_pred_price))
r2 = r2_score(y_test, stacked_pred_price)
dir_acc = np.mean(np.sign(y_test_ret) == np.sign(stacked_pred_ret))

print(f"\nStacking Ensemble — Held-out Test Results:")
print(f"  MAE  = ${mae:.2f}")
print(f"  RMSE = ${rmse:.2f}")
print(f"  R²   = {r2:.4f}")
print(f"  Directional Accuracy = {dir_acc:.1%}")

# ── Backtest: long when model predicts up, flat otherwise ─────────────────────
signal = np.where(stacked_pred_ret > 0, 1.0, 0.0)
strategy_daily = signal * y_test_ret
bh_daily = y_test_ret

strategy_cum = np.exp(np.cumsum(strategy_daily)) - 1
bh_cum = np.exp(np.cumsum(bh_daily)) - 1

sharpe = (strategy_daily.mean() / strategy_daily.std()) * np.sqrt(252)
bh_sharpe = (bh_daily.mean() / bh_daily.std()) * np.sqrt(252)
strat_mdd = max_drawdown(strategy_cum)
bh_mdd = max_drawdown(bh_cum)

print(f"\nBacktest (Long/Flat on test set):")
print(f"  Strategy return:  {strategy_cum[-1]:.1%}  |  Sharpe: {sharpe:.2f}  |  Max DD: {strat_mdd:.1%}")
print(f"  Buy & Hold:       {bh_cum[-1]:.1%}  |  Sharpe: {bh_sharpe:.2f}  |  Max DD: {bh_mdd:.1%}")

# ── Feature importance from best individual model ─────────────────────────────
best_name = min(cv_results, key=lambda k: cv_results[k]["RMSE"])
best_model = base_models[best_name]
best_model.fit(X_scaled[:split], y_ret[:split])

# ── Visualization (9 panels) ──────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 20))
fig.suptitle(
    "AAPL Advanced Stock Prediction — Stacking Ensemble",
    fontsize=17, fontweight="bold", y=0.99,
)
gs = fig.add_gridspec(4, 3, hspace=0.42, wspace=0.35)

# 1. Full history
ax = fig.add_subplot(gs[0, :2])
ax.plot(dates, close_series.reindex(dates), label="Actual", color="steelblue", lw=0.8, alpha=0.85)
ax.plot(dates_test, stacked_pred_price, label="Predicted (Stacking)", color="tomato", lw=1.1)
ax.axvline(dates[split], color="green", ls="--", lw=1.5, label="Train/Test split")
ax.set_title("Full Price History + Stacking Ensemble Predictions")
ax.set_ylabel("Price ($)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 2. Metrics summary
ax = fig.add_subplot(gs[0, 2])
ax.axis("off")
txt = (
    f"Stacking Ensemble\n"
    f"{'─'*30}\n"
    f"MAE:          ${mae:.2f}\n"
    f"RMSE:         ${rmse:.2f}\n"
    f"R\u00b2:           {r2:.4f}\n"
    f"Dir. Acc:     {dir_acc:.1%}\n\n"
    f"Backtest (test set)\n"
    f"{'─'*30}\n"
    f"Strategy ret: {strategy_cum[-1]:.1%}\n"
    f"Buy & Hold:   {bh_cum[-1]:.1%}\n"
    f"Sharpe:       {sharpe:.2f}  (B&H: {bh_sharpe:.2f})\n"
    f"Max DD:       {strat_mdd:.1%}  (B&H: {bh_mdd:.1%})\n"
)
ax.text(
    0.05, 0.97, txt, transform=ax.transAxes, fontsize=9.5,
    va="top", fontfamily="monospace",
    bbox=dict(boxstyle="round", facecolor="#eef3f8", alpha=0.95),
)
ax.set_title("Performance Summary")

# 3. Zoomed test period
ax = fig.add_subplot(gs[1, :2])
ax.plot(dates_test, y_test, label="Actual", color="steelblue", lw=1.5)
ax.plot(dates_test, stacked_pred_price, label="Predicted", color="tomato", ls="--", lw=1.5)
ax.fill_between(dates_test, y_test, stacked_pred_price, alpha=0.15, color="orange")
ax.set_title("Test Period: Actual vs Predicted Price")
ax.set_ylabel("Price ($)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 4. Equity curve
ax = fig.add_subplot(gs[1, 2])
ax.plot(dates_test, strategy_cum * 100, label=f"Strategy ({strategy_cum[-1]:.1%})", color="green", lw=1.5)
ax.plot(dates_test, bh_cum * 100, label=f"Buy & Hold ({bh_cum[-1]:.1%})", color="steelblue", lw=1.5, ls="--")
ax.axhline(0, color="black", lw=0.8)
ax.fill_between(dates_test, strategy_cum * 100, 0, alpha=0.15,
                where=strategy_cum > 0, color="green")
ax.fill_between(dates_test, strategy_cum * 100, 0, alpha=0.15,
                where=strategy_cum <= 0, color="red")
ax.set_title("Backtest Equity Curve")
ax.set_ylabel("Cumulative Return (%)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 5. Returns scatter
ax = fig.add_subplot(gs[2, 0])
ax.scatter(y_test_ret * 100, stacked_pred_ret * 100, alpha=0.2, s=6, color="purple")
lim = max(abs(y_test_ret).max(), abs(stacked_pred_ret).max()) * 100 * 1.05
ax.plot([-lim, lim], [-lim, lim], "r--", lw=1)
ax.axhline(0, color="gray", lw=0.5)
ax.axvline(0, color="gray", lw=0.5)
ax.set_title("Actual vs Predicted Returns")
ax.set_xlabel("Actual Return (%)")
ax.set_ylabel("Predicted Return (%)")
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.grid(True, alpha=0.3)

# 6. Residuals
ax = fig.add_subplot(gs[2, 1])
residuals = y_test - stacked_pred_price
ax.plot(dates_test, residuals, color="darkorange", lw=0.8)
ax.axhline(0, color="red", ls="--", lw=1)
ax.fill_between(dates_test, residuals, 0, alpha=0.2, color="orange")
ax.set_title(f"Price Residuals  (MAE=${mae:.2f})")
ax.set_ylabel("Error ($)")
ax.grid(True, alpha=0.3)

# 7. Rolling 60-day directional accuracy
ax = fig.add_subplot(gs[2, 2])
correct = (np.sign(y_test_ret) == np.sign(stacked_pred_ret)).astype(float)
rolling_dir = pd.Series(correct, index=dates_test).rolling(60).mean() * 100
ax.plot(dates_test, rolling_dir, color="teal", lw=1.5)
ax.axhline(50, color="red", ls="--", lw=1, label="Random (50%)")
ax.axhline(dir_acc * 100, color="green", ls=":", lw=1.2, label=f"Mean ({dir_acc:.1%})")
ax.set_title("Rolling 60-day Directional Accuracy")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(25, 85)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 8. Feature importance
ax = fig.add_subplot(gs[3, :2])
if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=feature_cols).nlargest(20)
    imp[::-1].plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(f"Top 20 Feature Importances  ({best_name})")
    ax.set_xlabel("Importance")
    ax.grid(True, alpha=0.3, axis="x")

# 9. Model comparison bar chart
ax = fig.add_subplot(gs[3, 2])
names = list(cv_results.keys()) + ["Stacking"]
dir_accs = [cv_results[n]["DirAcc"] * 100 for n in cv_results] + [dir_acc * 100]
colors = ["steelblue"] * len(cv_results) + ["gold"]
bars = ax.bar(range(len(names)), dir_accs, color=colors, edgecolor="white", width=0.6)
ax.axhline(50, color="red", ls="--", lw=1, label="Random baseline")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Directional Accuracy (%)")
ax.set_title("Model Comparison (CV Dir. Acc)")
ax.set_ylim(40, 80)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, dir_accs):
    ax.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
        f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
    )

plt.savefig("stock_prediction.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved \u2192 stock_prediction.png")
