import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

#csv file
file_path = r".csv" #insert glass node or other data source .csv file path
df = pd.read_csv(file_path)

df['timestamp'] = pd.to_datetime(df['timestamp'])

#plot graph
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['value'], label='Percent Balance on Exchanges', color='blue', linewidth=2)

#customize graph
plt.title('Ethereum Balance on Exchanges', fontsize=16)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.legend()
plt.grid(True)
plt.tight_layout()

#data & source labs
data_label_x = df['timestamp'].min() + pd.DateOffset(days=-300)
plt.text(data_label_x, df['value'].min() - 0.05, 'Data as of 18 June 2024', fontsize=10, ha='left')

source_label_x = df['timestamp'].max() - pd.DateOffset(days=10)
plt.text(source_label_x, df['value'].min() - 0.05, 'Source: Glassnode', fontsize=10, ha='right')

plt.show()
