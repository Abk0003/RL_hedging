import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import yfinance as yf
import numpy as np
import pandas as pd

data = yf.download(tickers=["SPY","QQQ","^VIX","^TNX","^FVX","^IRX","DX-Y.NYB","TLT","IEF","LQD","GC=F","CL=F"],start="2000-01-01",end="2025-12-31", auto_adjust=True,
    group_by="ticker")
print(data.isna().sum().sort_values(ascending=False))
print(data.isna().any(axis=1).sum())
print(data.shape)
print(data.columns)





