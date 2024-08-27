import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates

file_path = r".csv" #insert glass node or other data source .csv file path

data = pd.read_csv(file_path)

#normalize data
data['DATE'] = pd.to_datetime(data['DATE'])
data['APY'] = pd.to_numeric(data['APY'], errors='coerce')
data = data.dropna(subset=['APY'])
sns.set(style='darkgrid')
fig, ax = plt.subplots(figsize=(14, 7))

#APY plot
sns.lineplot(x='DATE', y='APY', data=data, ax=ax, color='blue', linewidth=2.5)

#plot customization
ax.fill_between(data['DATE'], data['APY'], color='blue', alpha=0.3)
ax.set_title('2024 Ethena APY', fontsize=20)
ax.set_ylabel('APY', fontsize=14)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y/100)))
ax.set_ylim(0, 60)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%B'))
ax.set_xlabel('')

#data source and date labels
data_source = "Source: DefiLlama"
data_date = "Data as of August 4, 2024"
plt.figtext(0.10, 0.01, data_source, horizontalalignment='left', fontsize=12)
plt.figtext(0.88, 0.01, data_date, horizontalalignment='right', fontsize=12)

#latest data point
latest_date = data['DATE'].max()
latest_apy = data.loc[data['DATE'] == latest_date, 'APY'].values[0]

ax.annotate(f'{latest_apy:.2f}%', 
                xy=(datetime.today(), latest_apy), 
                xytext=(-30, 15),
                textcoords='offset points', 
                arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()
