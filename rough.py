import yfinance as yf
import numpy as np
import pandas as pd

SKW = yf.download("^SKEW", start="2005-01-01", end="2025-12-31")
skew = SKW["Close"]

import matplotlib.pyplot as plt
plt.plot(skew)
plt.title("Raw CBOE SKEW index, full history")
plt.show()