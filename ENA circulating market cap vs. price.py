import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

#api
api_url = "https://api.coingecko.com/api/v3/coins/ethena/market_chart"
params = {
    'vs_currency': 'usd',
    'days': '365'
}

response = requests.get(api_url, params=params)
data = response.json()

#price and market cap data
price_data = data['prices']
market_cap_data = data['market_caps']

#convert data into DataFrames
price_df = pd.DataFrame(price_data, columns=['timestamp', 'price'])
market_cap_df = pd.DataFrame(market_cap_data, columns=['timestamp', 'market_cap'])

#convert timestamps to datetime
price_df['Date'] = pd.to_datetime(price_df['timestamp'], unit='ms')
market_cap_df['Date'] = pd.to_datetime(market_cap_df['timestamp'], unit='ms')

price_df = price_df.drop(columns=['timestamp'])
market_cap_df = market_cap_df.drop(columns=['timestamp'])

#merge data into single dataframe
df = pd.merge(price_df, market_cap_df, on='Date', how='outer')

#plot market cap
sns.set(style="whitegrid")

fig, ax1 = plt.subplots(figsize=(14, 7))

ax1.fill_between(df['Date'], df['market_cap'], color='tab:green', alpha=0.3, label='Market Cap')
ax1.set_xlabel('', fontsize=14, color='black')
ax1.set_ylabel('Market Cap (USD)', color='black', fontsize=14)
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, linestyle='--', alpha=0.7)

ax1.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e9:.1f}B'))

#plot price
ax2 = ax1.twinx()
ax2.plot(df['Date'], df['price'], color='tab:blue', label='Price', linestyle='-', linewidth=2)
ax2.set_ylabel('Price (USD)', color='black', fontsize=14)
ax2.tick_params(axis='y', labelcolor='black')

fig.autofmt_xdate()
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.title('ENA Circulating Market Cap vs Price', fontsize=16, color='black')

ax1.legend(loc='upper right', bbox_to_anchor=(.99, .91), fontsize=12)
ax2.legend(loc='upper right', bbox_to_anchor=(.99, .99), fontsize=12)

plt.figtext(0.01, -.02, 'Source: CoinGecko', ha='left', fontsize=12, color='black')
plt.figtext(0.99, -.02, 'Data as of August 5, 2024', ha='right', fontsize=12, color='black')

plt.tight_layout()
plt.show()
