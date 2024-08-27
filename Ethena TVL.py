import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates

file_path = r".csv" #insert glass node or other data source .csv file path
data = pd.read_csv(file_path)

#normalize data
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
data['Ethena'] = pd.to_numeric(data['Ethena'], errors='coerce')
data = data.dropna(subset=['Ethena'])
data['Ethena'] = data['Ethena'] / 1e9

sns.set(style='darkgrid')
fig, ax = plt.subplots(figsize=(14, 7))

#TVL plot
sns.lineplot(x='Date', y='Ethena', data=data, ax=ax, color='blue', linewidth=2.5)

#plot customization
ax.fill_between(data['Date'], data['Ethena'], color='blue', alpha=0.3)
ax.set_title("Ethena TVL", fontsize=20)
ax.set_ylabel('TVL (in billions)', fontsize=14)
ax.set_ylim(0, data['Ethena'].max() * 1.1)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%B'))
ax.set_xlabel('')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:.1f}B'))

#data source and date labels
data_source = "Source: DefiLlama"
data_date = "Data as of August 4, 2024"
plt.figtext(0.10, 0.01, data_source, horizontalalignment='left', fontsize=12)
plt.figtext(0.88, 0.01, data_date, horizontalalignment='right', fontsize=12)

#latest data point
latest_date = data['Date'].max()
latest_tvl = data.loc[data['Date'] == latest_date, 'Ethena'].values[0]
ax.annotate(f'${latest_tvl:.2f}B', 
            xy=(latest_date, latest_tvl), 
            xytext=(-20, 15),
            textcoords='offset points', 
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=12,
            color='black')

plt.show()
