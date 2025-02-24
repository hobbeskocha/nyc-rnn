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

# # NYC: GRU Modeling

# ## Libraries

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error

from classes.electric_load_dataset import ElectricLoadDataset
from classes.helper_functions import init_weights
# -

torch.manual_seed(13)
random.seed(13)
np.random.seed(13)

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
