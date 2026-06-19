import yfinance as yf
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROLL_WINDOW = 504
MIN_PERIODS = 252

SPY   = yf.download("SPY",    start="2000-01-01", end="2025-12-31")
VIX   = yf.download("^VIX",   start="2005-01-01", end="2025-12-31")
TNX   = yf.download("^TNX",   start="2005-01-01", end="2025-12-31")
VIX3M = yf.download("^VIX3M", start="2005-01-01", end="2025-12-31")
SKW   = yf.download("^SKEW",  start="2005-01-01", end="2025-12-31")



SPY_returns = np.log(SPY["Close"] / SPY["Close"].shift(1))

def rolling_zscore(series, window=ROLL_WINDOW, min_periods=MIN_PERIODS):
    s = series.squeeze() if hasattr(series, "squeeze") else series
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std()
    return (s - mean) / (std + 1e-8)

rv_21 = SPY_returns.rolling(21).std() * np.sqrt(252)
rv_30 = SPY_returns.rolling(30).std() * np.sqrt(252)
rv_91 = SPY_returns.rolling(91).std() * np.sqrt(252)

vix    = VIX["Close"].squeeze()
tnx    = TNX["Close"].squeeze()
skew   = SKW["Close"].squeeze()
vix3m  = VIX3M["Close"].squeeze()

term_str = vix3m - vix

SPY_ret_5d  = SPY["Close"].pct_change(5).squeeze()
SPY_ret_10d = SPY["Close"].pct_change(10).squeeze()
SPY_ret_20d = SPY["Close"].pct_change(20).squeeze()

SMA_10 = (SPY["Close"] / SPY["Close"].rolling(10).mean() - 1).squeeze()
SMA_50 = (SPY["Close"] / SPY["Close"].rolling(50).mean() - 1).squeeze()
trend  = (SMA_10 / (SMA_50 + 1e-8) - 1)

VIX_change = VIX["Close"].pct_change().squeeze()


VIX_ma5  = VIX["Close"].rolling(5).mean().squeeze()
VIX_ma20 = VIX["Close"].rolling(20).mean().squeeze()
VIX_slope = VIX_ma5 - VIX_ma20

zscore_20 = ((SPY["Close"] - SPY["Close"].rolling(20).mean())
             / SPY["Close"].rolling(20).std()).squeeze()

realized_vol_change = rv_21.pct_change()
vix_vol = VIX["Close"].rolling(10).std().squeeze()

lag_1 = SPY_returns.shift(1)
lag_2 = SPY_returns.shift(2)
lag_5 = SPY_returns.shift(5)

rolling_max = SPY["Close"].cummax().squeeze()
drawdown = SPY["Close"].squeeze() / rolling_max - 1
max_dd_20 = drawdown.rolling(20).min()

ret_fwd = ((SPY["Close"].shift(-1) - SPY["Close"]) / SPY["Close"]).squeeze()

rv21_z   = rolling_zscore(rv_21)
rv30_z   = rolling_zscore(rv_30)
rv91_z   = rolling_zscore(rv_91)
term_z   = rolling_zscore(term_str)
vix_z    = rolling_zscore(vix)
vix3m_z  = rolling_zscore(vix3m)
skew_z   = rolling_zscore(skew)
tnx_z    = rolling_zscore(tnx)
ret5_z   = rolling_zscore(SPY_ret_5d)
ret10_z  = rolling_zscore(SPY_ret_10d)
ret20_z  = rolling_zscore(SPY_ret_20d)
sma10_z  = rolling_zscore(SMA_10)
sma50_z  = rolling_zscore(SMA_50)
trend_z  = rolling_zscore(trend)
vixchg_z = rolling_zscore(VIX_change)
vixslope_z = rolling_zscore(VIX_slope)
zscore20_z = rolling_zscore(zscore_20)
rvolchg_z  = rolling_zscore(realized_vol_change)
vixvol_z   = rolling_zscore(vix_vol)
lag1_z   = rolling_zscore(lag_1)
lag2_z   = rolling_zscore(lag_2)
lag5_z   = rolling_zscore(lag_5)
maxdd_z  = rolling_zscore(max_dd_20)

feature_dict = {"rv21": rv21_z,"rv30": rv30_z,"rv91": rv91_z,"term_str": term_z,"vix": vix_z,"vix3m": vix3m_z,"skew": skew_z,"tnx": tnx_z,"SPY_ret_5d": ret5_z,"SPY_ret_10d": ret10_z,"SPY_ret_20d": ret20_z,"SMA_10d": sma10_z,"SMA_50d": sma50_z,"trend": trend_z,"VIX_change": vixchg_z,"VIX_slope": vixslope_z,"zscore_20": zscore20_z,"realized_vol_change": rvolchg_z,"vix_vol": vixvol_z,"lag_1": lag1_z,"lag_2": lag2_z,"lag_5": lag5_z,"max_dd_20": maxdd_z,}

features = pd.concat(feature_dict.values(), axis=1)
features.columns = list(feature_dict.keys())
data = pd.concat([features, ret_fwd.rename("ret_fwd")], axis=1).dropna()
features = data.iloc[:, :-1]
ret_fwd = data.iloc[:, -1]

train_end = "2017-12-31"
val_end = "2021-01-01"

train_idx = features.index < train_end
valid_idx = (features.index >= train_end) & (features.index < val_end)
test_idx = features.index >= val_end

X_train = features[train_idx]
X_valid = features[valid_idx]
X_test = features[test_idx]

y_train = ret_fwd[train_idx]
y_valid = ret_fwd[valid_idx]
y_test = ret_fwd[test_idx]

X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_valid = torch.tensor(X_valid.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_valid = torch.tensor(y_valid.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

if __name__ == "__main__":
    print("X_train:", X_train.shape, "X_valid:", X_valid.shape, "X_test:", X_test.shape)
    print("NaNs in X_train:", torch.isnan(X_train).sum().item())
    print("NaNs in X_valid:", torch.isnan(X_valid).sum().item())
    print("NaNs in X_test:", torch.isnan(X_test).sum().item())
    feature_names = list(feature_dict.keys())
    for i, name in enumerate(feature_names):
        tr = X_train[:, i]
        te = X_test[:, i]
        print(f"{name:22s} train: mean={tr.mean():.2f} std={tr.std():.2f} | "
              f"test: mean={te.mean():.2f} std={te.std():.2f} max={te.max():.2f} min={te.min():.2f}")










