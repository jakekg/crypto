import requests
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt

def fetch_market_data(coin_id='bitcoin'):
    url = f'https://api.coingecko.com/api/v3/coins/markets'
    params = {
        'vs_currency': 'usd',
        'ids': coin_id
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Failed to fetch market data: {response.status_code}")
        print(response.text)
        return None

def fetch_historical_data(coin_id='bitcoin'):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {
        'vs_currency': 'usd',
        'days': '365'
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if 'prices' in data:
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        else:
            print("Key 'prices' not found in the response.")
            print(data)
            return pd.DataFrame()
    else:
        print(f"Failed to fetch historical data: {response.status_code}")
        print(response.text)
        return pd.DataFrame()

def technical_analysis(df):
    if not df.empty:
        df['SMA'] = ta.sma(df['price'], length=20)
    return df

def plot_data(df):
    if not df.empty:
        plt.figure(figsize=(14, 7))
        plt.plot(df['timestamp'], df['price'], label='Price')
        plt.plot(df['timestamp'], df['SMA'], label='20-Day SMA')
        plt.title('Bitcoin Price with SMA')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.show()
    else:
        print("No data to plot or data is empty.")

historical_data = fetch_historical_data()
if not historical_data.empty:
    ta_data = technical_analysis(historical_data)
    plot_data(ta_data)
else:
    print("No historical data available.")
