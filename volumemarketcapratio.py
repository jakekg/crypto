import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

def get_crypto_data(ticker_symbol):
    try:
        crypto_data = yf.Ticker(ticker_symbol)
        price = crypto_data.history(period='1d')['Close'][0]
        circulating_supply = crypto_data.info['circulatingSupply']
        volume_24h = crypto_data.info['volume24Hr']
        market_cap = price * circulating_supply
        return volume_24h, market_cap
    except Exception as e:
        print(f"An error occurred while fetching {ticker_symbol} data: {e}")
        return None, None

def plot_volume_market_cap_ratios(crypto_data):
    if any(volume is None or market_cap is None for volume, market_cap in crypto_data.values()):
        print("Data not available for all cryptocurrencies. Unable to plot.")
        return

    #volume/market cap ratios
    ratios = {symbol: (volume / market_cap) * 100 for symbol, (volume, market_cap) in crypto_data.items()}

    #ascending order
    sorted_ratios = {symbol: ratios[symbol] for symbol in sorted(ratios, key=lambda x: ratios[x], reverse=False)}

    #maximum ratio for data text labels
    max_ratio = max(sorted_ratios.values())

    #figure & axis
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title('Volume / Market Cap Ratios for Top 10 Cryptocurrencies', fontsize=16, fontweight='bold')

    #add gradient colors to bar graph
    colors = plt.cm.Blues(np.linspace(0.2, 1, len(sorted_ratios)))
    bars = ax.bar(sorted_ratios.keys(), sorted_ratios.values(), color=colors)

    #text labels with offset
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + max_ratio * 0.01, f"{yval:.2f}%", ha='center', va='bottom', fontsize=10)

    ax.set_xticklabels(sorted_ratios.keys(), rotation=45, ha='right', fontsize=12)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5)

    #data source & date
    ax.text(1, -0.15, "Source: Yahoo Finance", transform=ax.transAxes, ha='right', fontsize=11)
    ax.text(-0.025, -0.15, "Data as of 7 June 2024", transform=ax.transAxes, ha='left', fontsize=11)

    plt.tight_layout()
    plt.show()

ticker_symbols = {
    'Bitcoin': 'BTC-USD',
    'Ethereum': 'ETH-USD',
    'BNB': 'BNB-USD',
    'Solana': 'SOL-USD',
    'XRP': 'XRP-USD',
    'Dogecoin': 'DOGE-USD',
    'Toncoin': 'TON-USD',
    'Cardano': 'ADA-USD',
    'Shiba Inu': 'SHIB-USD',
    'Avalanche': 'AVAX-USD'
}

#crypto data
crypto_data = {symbol: get_crypto_data(ticker) for symbol, ticker in ticker_symbols.items()}

plot_volume_market_cap_ratios(crypto_data)
