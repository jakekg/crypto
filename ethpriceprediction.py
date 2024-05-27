import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

#data (can change cryptocurrency)
cryptocurrency = 'ETH-USD'
start = dt.datetime(2014, 1, 1)
end = dt.datetime.now()
data = yf.download(cryptocurrency, start=start, end=end)

#data normalized
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

#sequences function
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append((seq, label))
    return sequences

sequence_length = 60
sequences = create_sequences(scaled_data, sequence_length)

#sequences converted to PyTorch tensors
seq_tensors = torch.FloatTensor([s[0] for s in sequences])
label_tensors = torch.FloatTensor([s[1] for s in sequences])

#split dataset into training and validation sets
dataset = TensorDataset(seq_tensors, label_tensors)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#dataLoader for mini-batch training
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, hidden_layer_size),
                            torch.zeros(1, 1, hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

#model, loss function, & optimizer
model = LSTM()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#model training (can change # of epochs)
epochs = 100
for epoch in range(epochs):
    model.train()
    train_losses = []
    for seq, labels in train_loader:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, seq.size(0), model.hidden_layer_size),
                             torch.zeros(1, seq.size(0), model.hidden_layer_size))
        y_pred = model(seq)
        loss_value = criterion(y_pred, labels)
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss_value.item())
    
    model.eval()
    val_losses = []
    with torch.no_grad():
        for seq, labels in val_loader:
            model.hidden_cell = (torch.zeros(1, seq.size(0), model.hidden_layer_size),
                                 torch.zeros(1, seq.size(0), model.hidden_layer_size))
            y_pred = model(seq)
            loss_value = criterion(y_pred, labels)
            val_losses.append(loss_value.item())
    
    print(f'Epoch {epoch+1}, Training Loss: {np.mean(train_losses):.4f}, Validation Loss: {np.mean(val_losses):.4f}')

#model testing
test_start = dt.datetime.now() - dt.timedelta(days=100)
test_end = dt.datetime.now()
test_data = yf.download(cryptocurrency, start=test_start, end=test_end)
scaled_test_data = scaler.transform(test_data['Close'].values.reshape(-1, 1))

test_sequences = create_sequences(scaled_test_data, sequence_length)
test_sequences = [(torch.FloatTensor(seq), torch.FloatTensor(label)) for seq, label in test_sequences]

future_predictions = []
model.eval()
with torch.no_grad():
    for seq, _ in test_sequences:
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        y_pred = model(seq.unsqueeze(0))
        future_predictions.append(y_pred.item())

#extend predictions
last_seq = torch.FloatTensor(test_sequences[-1][0])
model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                     torch.zeros(1, 1, model.hidden_layer_size))

#future predictions (can change days_to_predict)
days_to_predict = 14
for _ in range(days_to_predict):
    with torch.no_grad():
        y_pred = model(last_seq.unsqueeze(0))
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
