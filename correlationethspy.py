import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

eth_ticker = "ETH-USD"
spy_ticker = "SPY"

#fetch data
eth_data = yf.download(eth_ticker, start="2023-01-01", end="2024-06-01")
spy_data = yf.download(spy_ticker, start="2023-01-01", end="2024-06-01")

eth_close = eth_data['Close']
spy_close = spy_data['Close']

#align data
data = pd.DataFrame({
    'ETH_Close': eth_close,
    'SPY_Close': spy_close
}).dropna()

#returns calculations
data['ETH_Returns'] = data['ETH_Close'].pct_change().dropna()
data['SPY_Returns'] = data['SPY_Close'].pct_change().dropna()

#rolling correlation calculations
window_size = 90  # Change this to set the window size for the rolling correlation
rolling_corr = data['ETH_Returns'].rolling(window=window_size).corr(data['SPY_Returns'])

#print
if not rolling_corr.empty:
    latest_corr = rolling_corr.iloc[-1]
    latest_date = rolling_corr.index[-1]
    print(f"Latest correlation between daily returns of {eth_ticker} and {spy_ticker}: {latest_corr:.2f}")
else:
    latest_corr = None
    latest_date = None
    print("Rolling correlation calculation resulted in an empty series")

#line chart
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")
ax = sns.lineplot(x=rolling_corr.index, y=rolling_corr)

plt.title(f"Rolling {window_size}-Day Correlation of ETH vs S&P 500", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Correlation", fontsize=14)

#labels
plt.text(0.00, -0.10, 'Data as of 20 June 2024', fontsize=10, ha='left', va='bottom', transform=ax.transAxes)
plt.text(0.95, -0.10, 'Source: Yahoo Finance', fontsize=10, ha='right', va='bottom', transform=ax.transAxes)

#latest data point
if latest_corr is not None:
    plt.scatter([latest_date], [latest_corr], color='red', zorder=5)
    plt.text(latest_date + pd.DateOffset(days=10), latest_corr, f'{latest_corr:.2f}', fontsize=12, color='red', ha='left', va='center')
    plt.plot([latest_date, latest_date + pd.DateOffset(days=10)], [latest_corr, latest_corr], color='red', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
