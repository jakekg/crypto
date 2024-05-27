import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter, LogLocator
import mplfinance as mpf
import seaborn as sns
import datetime as dt

currency = "USD"
metric = "Close"

start = dt.datetime(2019, 1, 1)
end = dt.datetime.now()

crypto = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'BNB-USD']
colnames = []

first = True

for ticker in crypto:
    data = yf.download(ticker, start=start, end=end)
    if first:
        combined = data[[metric]].copy()
        colnames.append(ticker.split('-')[0])
        combined.columns = colnames
        first = False
    else:
        combined = combined.join(data[[metric]])
        colnames.append(ticker.split('-')[0])
        combined.columns = colnames

plt.figure(figsize=(12, 8))
plt.yscale('log')

colors = ['b', 'orange', 'g', 'r', 'purple']
styles = ['-', '--', '-.', ':', '-']
for i, ticker in enumerate(crypto):
    plt.plot(combined[ticker.split('-')[0]], label=ticker.split('-')[0], color=colors[i], linestyle=styles[i])

plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.title('Top 5 Layer 1s Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True, which="both", ls="--")

ax = plt.gca()
ax.yaxis.set_major_formatter(LogFormatter())
ax.yaxis.set_minor_formatter(LogFormatter(minor_thresholds=(2, 0.4)))

ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=100))

plt.savefig('l1price.png', bbox_inches='tight')
plt.show()
