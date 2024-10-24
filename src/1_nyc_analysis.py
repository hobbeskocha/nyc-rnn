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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error
# -

torch.manual_seed(13)
random.seed(13)
np.random.seed(13)

# ## Import Data

# +
nyc_train = pd.read_csv("../data/nyc_ny_train_hourly_interpolated.csv", index_col= "UTC_Timestamp")
nyc_test = pd.read_csv("../data/nyc_ny_test_hourly.csv", index_col= "UTC_Timestamp")

nyc_train.columns = ["Actual_Load_MW", "Temperature_Fahrenheit"]
nyc_test.columns = ["Actual_Load_MW", "Temperature_Fahrenheit"]
# -

nyc_train.info()
nyc_train.head()

nyc_test.info()
nyc_test.head()


# ## LSTM Modeling

# ### Class Definitions

# +
class ElectricLoadDataset(Dataset):
    def __init__(self, df, seq_len = 24):
        self.seq_len = seq_len
        self.data = df.values

    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len, 0]
        return torch.tensor(x, dtype = torch.float32), torch.tensor(y, dtype = torch.float32)
    
class LSTMModel(nn.Module):
    def __init__(self, input, hidden, n_layers, dropout_prob = 0.1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden
        self.num_layers = n_layers
        self.lstm = nn.LSTM(
            input_size = input,
            hidden_size = hidden,
            num_layers = n_layers,
            batch_first = True,
            dropout = dropout_prob
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


# -

# ### Model Training

def init_weights(m):
    if isinstance(m, nn.Linear):
        # Apply Xavier initialization to Linear layers
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
        # Apply Xavier initialization to LSTM or GRU weights
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


# +
sequence_length = 24
batch_size = 32
input_size = 2
hidden_size = 50
num_layers = 2
dropout_probability = 0.2
epochs = 30

normalizer = MinMaxScaler(feature_range=(0, 1))
nyc_train_normalized = nyc_train.copy()
nyc_train_normalized = pd.DataFrame(normalizer.fit_transform(nyc_train_normalized),
                                     columns=nyc_train.columns,
                                     index=nyc_train.index)

train_elec_dataset = ElectricLoadDataset(nyc_train_normalized, sequence_length)
train_dataloader = DataLoader(train_elec_dataset, batch_size = batch_size, shuffle = False)

lstm = LSTMModel(input_size, hidden_size, num_layers, dropout_probability)
lstm.apply(init_weights)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm.parameters(), lr = 0.0001)
losses = list()
# -

lstm.train()
for epoch in range(epochs):
    for x, y in train_dataloader:
        x = x.view(x.size(0), sequence_length, input_size)
        optimizer.zero_grad()
        output = lstm(x)
        loss = criterion(output, y.view(-1, 1))
        loss.backward()
        nn.utils.clip_grad_norm_(lstm.parameters(), max_norm = 1.0)
        optimizer.step()

    print(f"Epoch {epoch + 1} with loss: {loss.item()}")
    losses.append(loss.item())

lstm_loss_df = pd.DataFrame({"epoch": list(range(epochs)), "loss": losses})
sns.lineplot(lstm_loss_df, x = "epoch", y = "loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title(f"LSTM Training Loss over {epochs} Epochs")
plt.show()

print("Average LSTM Train Loss", lstm_loss_df["loss"].mean())

# ### Model Inference

# +
nyc_test_normalized = nyc_test.copy()
nyc_test_normalized = nyc_test_normalized.ffill(axis = 0)
nyc_test_normalized = pd.DataFrame(normalizer.transform(nyc_test_normalized),
                                    columns=nyc_test.columns,
                                    index=nyc_test.index)

test_elec_dataset = ElectricLoadDataset(nyc_test_normalized, sequence_length)
test_dataloader = DataLoader(test_elec_dataset, batch_size = batch_size, shuffle  = False)
predictions, actuals = list(), list()
mse = nn.MSELoss()
total_mse = 0
n_samples = 0

lstm.eval()
with torch.no_grad():
    for seq, label in test_dataloader:
        seq = seq.view(seq.size(0), sequence_length, input_size)
        output = lstm(seq)

        predictions.append(output.numpy())
        actuals.append(label.numpy())
        
        loss = mse(output, label.view(-1, 1))
        total_mse += loss.item() * seq.size(0)
        n_samples += seq.size(0)

avg_mse = total_mse / n_samples
print("Test MSE: ", avg_mse)

predictions = np.concatenate(predictions, axis = 0)
actuals = np.concatenate(actuals, axis = 0)

# +
data_predictions = {"predictions": pd.Series(predictions.squeeze()),
                    "temperature": pd.Series(nyc_test_normalized["Temperature_Fahrenheit"].values[24:])}
data_predictions = pd.DataFrame(data_predictions)

data_actuals = {"actuals": pd.Series(actuals.squeeze()),
                    "temperature": pd.Series(nyc_test_normalized["Temperature_Fahrenheit"].values[24:])}
data_actuals = pd.DataFrame(data_actuals)

data_predictions = normalizer.inverse_transform(data_predictions)
data_actuals = normalizer.inverse_transform(data_actuals)

# +
data = {"Predicted_Load": data_predictions[:, 0],
        "Actual_Load": data_actuals[:, 0]}

model_predictions = pd.DataFrame(data)
model_predictions.index = nyc_test[24:].index
model_predictions
# -

model_predictions.isna().sum()

lstm_model_rmse = root_mean_squared_error(model_predictions["Actual_Load"], model_predictions["Predicted_Load"])
print(lstm_model_rmse)

nyc_predictions = pd.merge(nyc_test[24:], model_predictions, on = "UTC_Timestamp")
nyc_predictions

# +
nyc_predictions.index = pd.to_datetime(nyc_predictions.index)
nyc_predictions_daily = nyc_predictions.resample("D").sum()

sns.lineplot(nyc_predictions, x = nyc_predictions.index, y = "Predicted_Load", label = "Predicted")
sns.lineplot(nyc_predictions, x = nyc_predictions.index, y = "Actual_Load", label = "Actual", alpha = 0.7)
plt.xticks(rotation = 45)
plt.title("LSTM Load Demand Predictions vs Actuals")

plt.savefig("../artifacts/nyc-predicted-actual-line.png")
plt.show()
# -

nyc_predictions.drop(["Actual_Load_MW"], axis = 1, inplace = True)
nyc_predictions["month"] = nyc_predictions.index.month
nyc_predictions["hour"] = nyc_predictions.index.hour

# +
sns.boxplot(nyc_predictions, x = "month", y = "Predicted_Load").set(title = "NYC Predicted Load by Month")
plt.show()

sns.boxplot(nyc_predictions, x = "month", y = "Actual_Load").set(title = "NYC Actual Load by Month")
plt.show()

# +
sns.boxplot(nyc_predictions, x = "hour", y = "Predicted_Load").set(title = "NYC Predicted Load by Hour")
plt.show()

sns.boxplot(nyc_predictions, x = "hour", y = "Actual_Load").set(title = "NYC Actual Load by Hour")
plt.show()

# +
avg_peak_month_load = nyc_predictions.query("month == 7")[["Predicted_Load", "Actual_Load"]].mean()
print(avg_peak_month_load)

avg_peak_hourly_load = nyc_predictions.query("hour == 22")[["Predicted_Load", "Actual_Load"]].mean()
print(avg_peak_hourly_load)
# -

# ## XGBoost Modeling

# +
nyc_train_xgb = pd.read_csv("../data/nyc_ny_train_hourly_interpolated.csv", index_col= "UTC_Timestamp")
nyc_test_xgb = pd.read_csv("../data/nyc_ny_test_hourly.csv", index_col= "UTC_Timestamp")

nyc_train_xgb.columns = ["Actual_Load_MW", "Temperature_Fahrenheit"]
nyc_test_xgb.columns = ["Actual_Load_MW", "Temperature_Fahrenheit"]
# -

# ### Feature Engineering

sequence = 24
nyc_test_xgb.ffill(axis=0, inplace=True)

# +
xgb_X_train = nyc_train_xgb.copy()
xgb_y_train = nyc_train_xgb[["Actual_Load_MW"]]

for s in range(1, sequence + 1):
    xgb_X_train[f"load_lag_{s}"] = xgb_X_train[["Actual_Load_MW"]].shift(s)
    xgb_X_train[f"temp_lag_{s}"] = xgb_X_train[["Temperature_Fahrenheit"]].shift(s)

xgb_X_train = xgb_X_train.drop(["Actual_Load_MW", "Temperature_Fahrenheit"], axis = 1)
xgb_X_train.dropna(axis = 0, inplace = True)
xgb_y_train = xgb_y_train[24:]

# +
xgb_X_test = nyc_test_xgb.copy()
xgb_y_test = nyc_test_xgb[["Actual_Load_MW"]]

for s in range(1, sequence + 1):
    xgb_X_test[f"load_lag_{s}"] = xgb_X_test[["Actual_Load_MW"]].shift(s)
    xgb_X_test[f"temp_lag_{s}"] = xgb_X_test[["Temperature_Fahrenheit"]].shift(s)

xgb_X_test = xgb_X_test.drop(["Actual_Load_MW", "Temperature_Fahrenheit"], axis = 1)
xgb_X_test.dropna(axis = 0, inplace = True)
xgb_y_test = xgb_y_test[24:]
# -

print(xgb_X_train.shape)
print(xgb_y_train.shape)
print(xgb_X_test.shape)
print(xgb_y_test.shape)

normalizer_X = MinMaxScaler(feature_range=(0, 1))
xgb_X_train = pd.DataFrame(normalizer_X.fit_transform(xgb_X_train),
                            columns=xgb_X_train.columns, index=xgb_X_train.index)
xgb_X_test = pd.DataFrame(normalizer_X.transform(xgb_X_test),
                           columns=xgb_X_test.columns, index=xgb_X_test.index)

normalizer_y = MinMaxScaler(feature_range=(0, 1))
xgb_y_train = pd.DataFrame(normalizer_y.fit_transform(xgb_y_train),
                            columns=xgb_y_train.columns, index=xgb_y_train.index)
xgb_y_test = pd.DataFrame(normalizer_y.transform(xgb_y_test),
                           columns=xgb_y_test.columns, index=xgb_y_test.index)

# ### XGBoost Training

xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators = 100)
xgb_model.fit(xgb_X_train, xgb_y_train)

# +
xgb_y_pred = xgb_model.predict(xgb_X_test)
xgb_y_pred = xgb_y_pred.reshape(-1, 1)

xgb_rmse = root_mean_squared_error(normalizer_y.inverse_transform(xgb_y_test),
                                   normalizer_y.inverse_transform(xgb_y_pred))
print(xgb_rmse)

# +
xgb_y_viz = xgb_y_test.copy()
xgb_y_viz["Predicted_Load"] = normalizer_y.inverse_transform(xgb_y_pred)
xgb_y_viz["Actual_Load_MW"] = normalizer_y.inverse_transform(xgb_y_viz[["Actual_Load_MW"]])
xgb_y_viz.index = pd.to_datetime(xgb_y_viz.index)
xgb_y_viz_daily = xgb_y_viz.resample("D").sum()

sns.lineplot(xgb_y_viz, x = xgb_y_viz.index, y = "Predicted_Load", label = "Predicted")
sns.lineplot(xgb_y_viz, x = xgb_y_viz.index, y = "Actual_Load_MW", label = "Actual", alpha = 0.7)
plt.xticks(rotation = 45)

plt.savefig("../artifacts/xgb-predicted-actual-line.png")
plt.show()
# -

# ## GRU Modeling

# +
nyc_train = pd.read_csv("../data/nyc_ny_train_hourly_interpolated.csv", index_col= "UTC_Timestamp")
nyc_test = pd.read_csv("../data/nyc_ny_test_hourly.csv", index_col= "UTC_Timestamp")

nyc_train.columns = ["Actual_Load_MW", "Temperature_Fahrenheit"]
nyc_test.columns = ["Actual_Load_MW", "Temperature_Fahrenheit"]


# -

# ### Class Definitions

class GRUModel(nn.Module):
    def __init__(self, input, hidden, n_layers, dropout_prob = 0.1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden
        self.num_layers = n_layers
        self.gru = nn.GRU(
            input_size = input,
            hidden_size = hidden,
            num_layers = n_layers,
            batch_first = True,
            dropout = dropout_prob
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


# ### Model Training

# +
sequence_length = 24
batch_size = 32
input_size = 2
hidden_size = 50
num_layers = 2
dropout_probability = 0.2
epochs = 30

normalizer = MinMaxScaler(feature_range=(0, 1))
nyc_train_normalized = nyc_train.copy()
nyc_train_normalized = pd.DataFrame(normalizer.fit_transform(nyc_train_normalized),
                                     columns=nyc_train.columns,
                                     index=nyc_train.index)

train_elec_dataset = ElectricLoadDataset(nyc_train_normalized, sequence_length)
train_dataloader = DataLoader(train_elec_dataset, batch_size = batch_size, shuffle = False)

gru = GRUModel(input_size, hidden_size, num_layers, dropout_probability)
gru.apply(init_weights)
criterion = nn.MSELoss()
optimizer = optim.Adam(gru.parameters(), lr = 0.0001)
losses = list()
# -

gru.train()
for epoch in range(epochs):
    for x, y in train_dataloader:
        x = x.view(x.size(0), sequence_length, input_size)
        optimizer.zero_grad()
        output = gru(x)
        loss = criterion(output, y.view(-1, 1))
        loss.backward()
        nn.utils.clip_grad_norm_(gru.parameters(), max_norm = 1.0)
        optimizer.step()

    print(f"Epoch {epoch + 1} with loss: {loss.item()}")
    losses.append(loss.item())

gru_loss_df = pd.DataFrame({"epoch": list(range(epochs)), "loss": losses})
sns.lineplot(gru_loss_df, x = "epoch", y = "loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title(f"GRU Training Loss over {epochs} Epochs")
plt.show()

print("Average GRU Train Loss", gru_loss_df["loss"].mean())

# ### Model Inference

# +
nyc_test_normalized = nyc_test.copy()
nyc_test_normalized = nyc_test_normalized.ffill(axis = 0)
nyc_test_normalized = pd.DataFrame(normalizer.transform(nyc_test_normalized),
                                    columns=nyc_test.columns,
                                    index=nyc_test.index)

test_elec_dataset = ElectricLoadDataset(nyc_test_normalized, sequence_length)
test_dataloader = DataLoader(test_elec_dataset, batch_size = batch_size, shuffle  = False)
predictions, actuals = list(), list()
mse = nn.MSELoss()
total_mse = 0
n_samples = 0

gru.eval()
with torch.no_grad():
    for seq, label in test_dataloader:
        seq = seq.view(seq.size(0), sequence_length, input_size)
        output = gru(seq)

        predictions.append(output.numpy())
        actuals.append(label.numpy())
        
        loss = mse(output, label.view(-1, 1))
        total_mse += loss.item() * seq.size(0)
        n_samples += seq.size(0)

avg_mse = total_mse / n_samples
print("Average GRU Test MSE: ", avg_mse)

predictions = np.concatenate(predictions, axis = 0)
actuals = np.concatenate(actuals, axis = 0)

# +
data_predictions = {"predictions": pd.Series(predictions.squeeze()),
                    "temperature": pd.Series(nyc_test_normalized["Temperature_Fahrenheit"].values[24:])}
data_predictions = pd.DataFrame(data_predictions)

data_actuals = {"actuals": pd.Series(actuals.squeeze()),
                    "temperature": pd.Series(nyc_test_normalized["Temperature_Fahrenheit"].values[24:])}
data_actuals = pd.DataFrame(data_actuals)

data_predictions = normalizer.inverse_transform(data_predictions)
data_actuals = normalizer.inverse_transform(data_actuals)

# +
gru_data = {"Predicted_Load": data_predictions[:, 0],
        "Actual_Load": data_actuals[:, 0]}

gru_model_predictions = pd.DataFrame(gru_data)
gru_model_predictions.index = nyc_test[24:].index
gru_model_predictions
# -

gru_model_predictions.isna().sum()

gru_model_rmse = root_mean_squared_error(gru_model_predictions["Actual_Load"], gru_model_predictions["Predicted_Load"])
print(gru_model_rmse)

nyc_predictions = pd.merge(nyc_test[24:], gru_model_predictions, on = "UTC_Timestamp")
nyc_predictions

# +
nyc_predictions.index = pd.to_datetime(nyc_predictions.index)
nyc_predictions_daily = nyc_predictions.resample("D").sum()

sns.lineplot(nyc_predictions, x = nyc_predictions.index, y = "Predicted_Load", label = "Predicted")
sns.lineplot(nyc_predictions, x = nyc_predictions.index, y = "Actual_Load", label = "Actual", alpha = 0.7)
plt.xticks(rotation = 45)
plt.title("GRU Load Demand Predictions vs Actuals")

plt.savefig("../artifacts/gru-predicted-actual-line.png")
plt.show()
