import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

#data
cryptocurrency = 'ETH-USD'
start = dt.datetime(2014, 1, 1)
end = dt.datetime.now()
data = yf.download(cryptocurrency, start=start, end=end)

#data normalized
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

#create sequences function
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append((seq, label))
    return sequences

sequence_length = 60
sequences = create_sequences(scaled_data, sequence_length)

#LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

#model, loss function, & optimizer
model = LSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#model training
epochs = 100
for epoch in range(epochs):
    for seq, labels in sequences:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))

        seq = torch.FloatTensor(seq)
        labels = torch.FloatTensor([labels])

        y_pred = model(seq)
        loss_value = criterion(y_pred, labels)
        loss_value.backward()
        optimizer.step()
        
    print(f'Epoch {epoch+1}, Loss: {loss_value.item()}')

#model testing
test_start = dt.datetime.now() - dt.timedelta(days=100)
test_end = dt.datetime.now()
test_data = yf.download(cryptocurrency, start=test_start, end=test_end)
scaled_test_data = scaler.transform(test_data['Close'].values.reshape(-1, 1))
test_sequences = create_sequences(scaled_test_data, sequence_length)

future_predictions = []
model.eval()
with torch.no_grad():
    for seq, _ in test_sequences:
        seq = torch.FloatTensor(seq)
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        y_pred = model(seq)
        future_predictions.append(y_pred.item())

#extend predictions
last_seq = torch.FloatTensor(test_sequences[-1][0])
model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                     torch.zeros(1, 1, model.hidden_layer_size))

#future
days_to_predict = 14
for _ in range(days_to_predict):
    with torch.no_grad():
        y_pred = model(last_seq)
        future_predictions.append(y_pred.item())
        last_seq = torch.cat((last_seq[1:], y_pred.view(1, 1)))

#inverse transform predictions
predicted_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

#plot results
plt.figure(figsize=(14, 5))
plt.plot(test_data.index[sequence_length:], test_data['Close'][sequence_length:], color='black', label='Actual Prices')
plt.plot(test_data.index[sequence_length:], predicted_prices[:len(test_data.index) - sequence_length], color='green', label='Predicted Prices')
future_dates = pd.date_range(test_data.index[-1] + pd.Timedelta(days=1), periods=days_to_predict)
plt.plot(future_dates, predicted_prices[len(test_data.index) - sequence_length:], color='red', linestyle='dashed', label='Future Predictions')
plt.title('Ethereum Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (in USD)')
plt.legend()
plt.show()
