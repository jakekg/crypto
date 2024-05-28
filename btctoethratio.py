import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime

#fetch historical data
def get_historical_data(symbol, start_time, end_time):
    data = yf.download(symbol, start=start_time, end=end_time)
    data.reset_index(inplace=True)
    data['Date'] = data['Date'].dt.date
    return data['Date'].tolist(), data['Adj Close'].tolist()

#historical data for btc and eth (can change dates for different timeline)
btc_dates, btc_prices = get_historical_data('BTC-USD', '2018-01-01', '2024-05-27')
eth_dates, eth_prices = get_historical_data('ETH-USD', '2018-01-01', '2024-05-27')

#convert lists to pandas series
btc_series = pd.Series(btc_prices, index=pd.to_datetime(btc_dates))
eth_series = pd.Series(eth_prices, index=pd.to_datetime(eth_dates))

#align series to same date index
btc_aligned, eth_aligned = btc_series.align(eth_series, join='inner')

#dataframe
df = pd.DataFrame({
    'Date': btc_aligned.index.date,
    'BTC_Price': btc_aligned.values,
    'ETH_Price': eth_aligned.values
})

#btc to eth ratio
df['BTC_ETH_Ratio'] = df['BTC_Price'] / df['ETH_Price']

#seaborn style
sns.set(style="whitegrid")

#figure and a set of subplots
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Date'], df['BTC_ETH_Ratio'], label='BTC/ETH Price Ratio', color='purple', linewidth=2)

#major and minor ticks formatting (change from year/month/weekday/day when changing timeline of chart)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
fig.autofmt_xdate()

#labels and title
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('BTC to ETH Ratio', fontsize=10)
ax.set_title('Historical BTC to ETH Price Ratio', fontsize=14)
plt.text(1.0, -.2, 'Source: Yahoo Finance', transform=ax.transAxes, ha='right', va='center', fontsize=9, color='gray')
sns.despine()
plt.show()
