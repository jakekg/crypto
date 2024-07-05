import requests

def get_eth_30d_avg_volume():
    #api endpoint
    url = 'https://api.coingecko.com/api/v3/coins/ethereum/market_chart'
    
    #api parameters
    params = {
        'vs_currency': 'usd',
        'days': '30',
        'interval': 'daily'
    }
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            volumes = [entry[1] for entry in data['total_volumes']]
            avg_volume_30d = sum(volumes) / len(volumes)
            return avg_volume_30d
        else:
            print(f"Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

#30-day average ETH trading volume
eth_30d_avg_volume = get_eth_30d_avg_volume()
if eth_30d_avg_volume is not None:
    print(f"The 30-day average trading volume of Ethereum is ${eth_30d_avg_volume:.2f}")
else:
    print("Failed to retrieve data.")
