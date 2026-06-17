import yfinance as yf
import numpy as np
import pandas as pd
import math
import torch
from torch.fx.experimental.migrate_gradual_types.constraint_transformation import valid_index

SPY = yf.download("SPY", start="2005-01-01", end="2025-12-31")
VIX = yf.download("^VIX", start="2005-01-01", end="2025-12-31")
TNX = yf.download("^TNX", start="2005-01-01", end="2025-12-31")
VIX3M = yf.download("^VIX3M", start="2005-01-01", end="2025-12-31")
SKW = yf.download("^SKEW", start="2005-01-01", end="2025-12-31")

SPY_returns = np.log(SPY["Close"] / SPY["Close"].shift(1))

rv_21 = SPY_returns.rolling(21).std().dropna() * np.sqrt(252)
rv_30 = SPY_returns.rolling(30).std().dropna() * np.sqrt(252)
rv_91 = SPY_returns.rolling(91).std().dropna() * np.sqrt(252)
vix = VIX["Close"]
tnx = TNX["Close"]
skew = SKW["Close"]
vix3m = VIX3M["Close"]
term_str = pd.DataFrame({"term_str": vix3m.squeeze() - vix.squeeze()})
SPY_ret_5d = SPY["Close"].pct_change(5)
SPY_ret_10d = SPY["Close"].pct_change(10)
SPY_ret_20d = SPY["Close"].pct_change(20)
SMA_10 = SPY["Close"].rolling(10).mean()
SMA_50 = SPY["Close"].rolling(50).mean()
trend = SMA_10 / SMA_50 - 1
VIX_change = VIX["Close"].pct_change()
VIX_ma5 = VIX["Close"].rolling(5).mean()
VIX_ma20 = VIX["Close"].rolling(20).mean()
VIX_slope = pd.DataFrame({"VIX_slope": VIX_ma5.squeeze() - VIX_ma20.squeeze()})
zscore_20 = (SPY["Close"] - SPY["Close"].rolling(20).mean()) / SPY["Close"].rolling(20).std()
realized_vol_change = rv_21.pct_change()
vix_vol = VIX["Close"].rolling(10).std()
lag_1 = SPY_returns.shift(1)
lag_2 = SPY_returns.shift(2)
lag_5 = SPY_returns.shift(5)
rolling_max = SPY["Close"].cummax()
drawdown = SPY["Close"] / rolling_max - 1
max_dd_20 = drawdown.rolling(20).min()
ret_fwd = ((SPY["Close"].shift(-1) - SPY["Close"])/SPY["Close"])

rv_21.columns = ["rv21"]
rv_30.columns = ["rv30"]
rv_91.columns = ["rv91"]
term_str.columns = ["term_str"]
vix.columns = ["vix"]
vix3m.columns = ["vix3m"]
skew.columns = ["skew"]
tnx.columns = ["tnx"]
SPY_ret_5d.columns = ["SPY_ret_5d"]
SPY_ret_10d.columns = ["SPY_ret_10d"]
SPY_ret_20d.columns = ["SPY_ret_20d"]
SMA_10.columns = ["SMA_10d"]
SMA_50.columns = ["SMA_50d"]
trend.columns = ["trend"]
VIX_change.columns = ["VIX_change"]
VIX_ma5.columns = ["VIX_ma5"]
VIX_ma20.columns = ["VIX_ma20"]
VIX_slope.columns = ["VIX_slope"]
zscore_20.columns = ["zscore_20"]
realized_vol_change.columns = ["realized_vol_change"]
vix_vol.columns = ["vix_vol"]
lag_1.columns = ["lag_1"]
lag_2.columns = ["lag_2"]
lag_5.columns = ["lag_5"]
max_dd_20.columns = ["max_dd_20"]

features = pd.concat([rv_21, rv_30, rv_91, term_str, vix, vix3m, skew, tnx,SPY_ret_5d,SPY_ret_10d,SPY_ret_20d,SMA_10,SMA_50,trend,VIX_change,VIX_ma5,VIX_slope,VIX_ma20,zscore_20,realized_vol_change,
                      vix_vol,lag_1,lag_2,lag_5,max_dd_20],axis=1)

data = pd.concat([features, ret_fwd], axis=1).dropna()
features = data.iloc[:, :-1]
ret_fwd = data.iloc[:, -1]

train_end = "2021-12-31"
val_end = "2023-01-01"

train_idx = features.index < train_end
valid_idx = (features.index >= train_end) & (features.index < val_end)
test_idx = features.index >= val_end

X_train = features[train_idx]
X_valid = features[valid_idx]
X_test = features[test_idx]

y_train = ret_fwd[train_idx]
y_valid = ret_fwd[valid_idx]
y_test = ret_fwd[test_idx]

mu = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mu)/(std + 1e-8)
X_valid = (X_valid - mu)/(std + 1e-8)
X_test  = (X_test  - mu)/(std + 1e-8)

X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_valid = torch.tensor(X_valid.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_valid = torch.tensor(y_valid.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)









