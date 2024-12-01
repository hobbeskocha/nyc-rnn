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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error

from classes.electric_load_dataset import ElectricLoadDataset
from classes.model_lstm import LSTMModel
from classes.model_gru import GRUModel
from classes.helper_functions import is_stationary, init_weights
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

# ## SARIMA Modeling

# +
nyc_train = pd.read_csv("../data/nyc_ny_train_hourly_interpolated.csv", index_col= "UTC_Timestamp")
nyc_test = pd.read_csv("../data/nyc_ny_test_hourly.csv", index_col= "UTC_Timestamp")

nyc_train.columns = ["Actual_Load_MW", "Temperature_Fahrenheit"]
nyc_test.columns = ["Actual_Load_MW", "Temperature_Fahrenheit"]
# -

# ### Check Stationarity and plot ACF/PACF

print(is_stationary(nyc_train["Actual_Load_MW"], significance=0.05))
print("\n")
print(is_stationary(nyc_train["Temperature_Fahrenheit"], significance=0.05))

plot_acf(nyc_train["Actual_Load_MW"]).show()
plot_pacf(nyc_train["Actual_Load_MW"]).show()

plot_acf(nyc_train["Temperature_Fahrenheit"]).show()
plot_pacf(nyc_train["Temperature_Fahrenheit"]).show()

# ## LSTM Modeling

# +
nyc_train = pd.read_csv("../data/nyc_ny_train_hourly_interpolated.csv", index_col= "UTC_Timestamp")
nyc_test = pd.read_csv("../data/nyc_ny_test_hourly.csv", index_col= "UTC_Timestamp")

nyc_train.columns = ["Actual_Load_MW", "Temperature_Fahrenheit"]
nyc_test.columns = ["Actual_Load_MW", "Temperature_Fahrenheit"]
# -

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

lstm = LSTMModel(input_size, hidden_size, num_layers, dropout_probability)
lstm.apply(init_weights)
lstm_criterion = nn.MSELoss()
lstm_optimizer = optim.Adam(lstm.parameters(), lr = 0.0001)
lstm_losses = list()
# -

lstm.train()
for epoch in range(epochs):
    for x, y in train_dataloader:
        x = x.view(x.size(0), sequence_length, input_size)
        lstm_optimizer.zero_grad()
        output = lstm(x)
        loss = lstm_criterion(output, y.view(-1, 1))
        loss.backward()
        nn.utils.clip_grad_norm_(lstm.parameters(), max_norm = 1.0)
        lstm_optimizer.step()

    print(f"Epoch {epoch + 1} with loss: {loss.item()}")
    lstm_losses.append(loss.item())

lstm_loss_df = pd.DataFrame({"epoch": list(range(epochs)), "loss": lstm_losses})
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
lstm_predictions, lstm_actuals = list(), list()
lstm_mse = nn.MSELoss()
lstm_total_mse = 0
lstm_n_samples = 0

lstm.eval()
with torch.no_grad():
    for seq, label in test_dataloader:
        seq = seq.view(seq.size(0), sequence_length, input_size)
        output = lstm(seq)

        lstm_predictions.append(output.numpy())
        lstm_actuals.append(label.numpy())
        
        loss = lstm_mse(output, label.view(-1, 1))
        lstm_total_mse += loss.item() * seq.size(0)
        lstm_n_samples += seq.size(0)

lstm_avg_mse = lstm_total_mse / lstm_n_samples
print("Test MSE: ", lstm_avg_mse)

lstm_predictions = np.concatenate(lstm_predictions, axis = 0)
lstm_actuals = np.concatenate(lstm_actuals, axis = 0)

# +
lstm_data_predictions = {"predictions": pd.Series(lstm_predictions.squeeze()),
                    "temperature": pd.Series(nyc_test_normalized["Temperature_Fahrenheit"].values[24:])}
lstm_data_predictions = pd.DataFrame(lstm_data_predictions)

lstm_data_actuals = {"actuals": pd.Series(lstm_actuals.squeeze()),
                    "temperature": pd.Series(nyc_test_normalized["Temperature_Fahrenheit"].values[24:])}
lstm_data_actuals = pd.DataFrame(lstm_data_actuals)

lstm_data_predictions = normalizer.inverse_transform(lstm_data_predictions)
lstm_data_actuals = normalizer.inverse_transform(lstm_data_actuals)

# +
lstm_data = {"Predicted_Load": lstm_data_predictions[:, 0],
        "Actual_Load": lstm_data_actuals[:, 0]}

lstm_model_predictions = pd.DataFrame(lstm_data)
lstm_model_predictions.index = nyc_test[24:].index
lstm_model_predictions
# -

lstm_model_predictions.isna().sum()

lstm_model_rmse = root_mean_squared_error(lstm_model_predictions["Actual_Load"], lstm_model_predictions["Predicted_Load"])
print(lstm_model_rmse)

lstm_nyc_predictions = pd.merge(nyc_test[24:], lstm_model_predictions, on = "UTC_Timestamp")
lstm_nyc_predictions

# +
lstm_nyc_predictions.index = pd.to_datetime(lstm_nyc_predictions.index)
lstm_nyc_predictions_daily = lstm_nyc_predictions.resample("D").sum()

sns.lineplot(lstm_nyc_predictions, x = lstm_nyc_predictions.index, y = "Predicted_Load", label = "Predicted")
sns.lineplot(lstm_nyc_predictions, x = lstm_nyc_predictions.index, y = "Actual_Load", label = "Actual", alpha = 0.7)
plt.xticks(rotation = 45)
plt.title("LSTM Load Demand Predictions vs Actuals")

plt.savefig("../artifacts/nyc-predicted-actual-line.png")
plt.show()
# -

lstm_nyc_predictions.drop(["Actual_Load_MW"], axis = 1, inplace = True)
lstm_nyc_predictions["month"] = lstm_nyc_predictions.index.month
lstm_nyc_predictions["hour"] = lstm_nyc_predictions.index.hour

# +
sns.boxplot(lstm_nyc_predictions, x = "month", y = "Predicted_Load").set(title = "NYC LSTM Predicted Load by Month")
plt.show()

sns.boxplot(lstm_nyc_predictions, x = "month", y = "Actual_Load").set(title = "NYC LSTM Actual Load by Month")
plt.show()

# +
sns.boxplot(lstm_nyc_predictions, x = "hour", y = "Predicted_Load").set(title = "NYC LSTM Predicted Load by Hour")
plt.show()

sns.boxplot(lstm_nyc_predictions, x = "hour", y = "Actual_Load").set(title = "NYC LSTM Actual Load by Hour")
plt.show()

# +
avg_peak_month_load = lstm_nyc_predictions.query("month == 7")[["Predicted_Load", "Actual_Load"]].mean()
print(avg_peak_month_load)

avg_peak_hourly_load = lstm_nyc_predictions.query("hour == 22")[["Predicted_Load", "Actual_Load"]].mean()
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

# ### XGBoost Inference

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
plt.title("XGBoost Predicted vs Actual Demand")

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
gru_criterion = nn.MSELoss()
gru_optimizer = optim.Adam(gru.parameters(), lr = 0.0001)
gru_losses = list()
# -

gru.train()
for epoch in range(epochs):
    for x, y in train_dataloader:
        x = x.view(x.size(0), sequence_length, input_size)
        gru_optimizer.zero_grad()
        output = gru(x)
        loss = gru_criterion(output, y.view(-1, 1))
        loss.backward()
        nn.utils.clip_grad_norm_(gru.parameters(), max_norm = 1.0)
        gru_optimizer.step()

    print(f"Epoch {epoch + 1} with loss: {loss.item()}")
    gru_losses.append(loss.item())

gru_loss_df = pd.DataFrame({"epoch": list(range(epochs)), "loss": gru_losses})
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
gru_predictions, gru_actuals = list(), list()
gru_mse = nn.MSELoss()
gru_total_mse = 0
gru_n_samples = 0

gru.eval()
with torch.no_grad():
    for seq, label in test_dataloader:
        seq = seq.view(seq.size(0), sequence_length, input_size)
        output = gru(seq)

        gru_predictions.append(output.numpy())
        gru_actuals.append(label.numpy())
        
        loss = gru_mse(output, label.view(-1, 1))
        gru_total_mse += loss.item() * seq.size(0)
        gru_n_samples += seq.size(0)

gru_avg_mse = gru_total_mse / gru_n_samples
print("Average GRU Test MSE: ", gru_avg_mse)

gru_predictions = np.concatenate(gru_predictions, axis = 0)
gru_actuals = np.concatenate(gru_actuals, axis = 0)

# +
gru_data_predictions = {"predictions": pd.Series(gru_predictions.squeeze()),
                    "temperature": pd.Series(nyc_test_normalized["Temperature_Fahrenheit"].values[24:])}
gru_data_predictions = pd.DataFrame(gru_data_predictions)

gru_data_actuals = {"actuals": pd.Series(gru_actuals.squeeze()),
                    "temperature": pd.Series(nyc_test_normalized["Temperature_Fahrenheit"].values[24:])}
gru_data_actuals = pd.DataFrame(gru_data_actuals)

gru_data_predictions = normalizer.inverse_transform(gru_data_predictions)
gru_data_actuals = normalizer.inverse_transform(gru_data_actuals)

# +
gru_data = {"Predicted_Load": gru_data_predictions[:, 0],
        "Actual_Load": gru_data_actuals[:, 0]}

gru_model_predictions = pd.DataFrame(gru_data)
gru_model_predictions.index = nyc_test[24:].index
gru_model_predictions
# -

gru_model_predictions.isna().sum()

gru_model_rmse = root_mean_squared_error(gru_model_predictions["Actual_Load"], gru_model_predictions["Predicted_Load"])
print(gru_model_rmse)

gru_nyc_predictions = pd.merge(nyc_test[24:], gru_model_predictions, on = "UTC_Timestamp")
gru_nyc_predictions

# +
gru_nyc_predictions.index = pd.to_datetime(gru_nyc_predictions.index)
gru_nyc_predictions_daily = gru_nyc_predictions.resample("D").sum()

sns.lineplot(gru_nyc_predictions, x = gru_nyc_predictions.index, y = "Predicted_Load", label = "Predicted")
sns.lineplot(gru_nyc_predictions, x = gru_nyc_predictions.index, y = "Actual_Load", label = "Actual", alpha = 0.7)
plt.xticks(rotation = 45)
plt.title("GRU Load Demand Predictions vs Actuals")

plt.savefig("../artifacts/gru-predicted-actual-line.png")
plt.show()
