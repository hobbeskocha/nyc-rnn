# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: rnn
#     language: python
#     name: python3
# ---

# # NYC: Modeling

# ## Libraries

# +
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
# -

# ## Import Data

# +
nyc_train = pd.read_csv("../data/nyc_ny_train_hourly_interpolated.csv", index_col= "UTC_Timestamp")
nyc_test = pd.read_csv("../data/nyc_ny_test_hourly.csv", index_col= "UTC_Timestamp")

nyc_train.columns = ["Actual_Load_MW"]
nyc_test.columns = ["Actual_Load_MW"]
# -

nyc_train.head()

nyc_test.head()


# ## Data Processing

# ### Class Definitions

# +
class ElectricLoadDataset(Dataset):
    def __init__(self, df, seq_len = 24):
        self.seq_len = seq_len
        self.data = df["Actual_Load_MW"].values.reshape(-1, 1)
        self.normalizer = MinMaxScaler(feature_range=(0, 1))
        self.data = self.normalizer.fit_transform(self.data)

    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return torch.tensor(x, dtype = torch.float32), torch.tensor(y, dtype = torch.float32)
    
class LSTMModel(nn.Module):
    def __init__(self, input, hidden, n_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden
        self.num_layers = n_layers
        self.lstm = nn.LSTM(
            input_size = input,
            hidden_size = hidden,
            num_layers = n_layers,
            batch_first = True
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# -

# ### Model Training

# +
sequence_length = 24
batch_size = 2
input_size = 1

hidden_size = 16
num_layers = 2
epochs = 5

# +
train_elec_dataset = ElectricLoadDataset(nyc_train, sequence_length)
train_dataloader = DataLoader(train_elec_dataset, batch_size = batch_size, shuffle = False)

lstm = LSTMModel(input_size, hidden_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm.parameters(), lr = 0.001)

for epoch in range(epochs):
    for x, y in train_dataloader:
        x = x.view(x.size(0), sequence_length, input_size)
        optimizer.zero_grad()
        output = lstm(x)
        loss = criterion(output, y.view(-1, 1))
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} with loss: {loss.item()}")
