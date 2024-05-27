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

combined = combined.pct_change().corr(method="pearson")

sns.heatmap(combined, annot=True, cmap="coolwarm")

plt.savefig('l1heatmap.png', bbox_inches='tight')
plt.show()