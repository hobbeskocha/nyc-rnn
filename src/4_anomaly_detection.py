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

# # NYC: Anomaly Detection

# ## Libraries

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import random
import xgboost as xgb
import shap
import optuna

from imblearn.combine import SMOTETomek

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
# -

random.seed(13)
np.random.seed(13)
tf.random.set_seed(13)

# ## Load data

# +
train_data = pd.read_csv("../data/nyc_ny_train_hourly_interpolated.csv", index_col= "UTC_Timestamp")
test_data = pd.read_csv("../data/nyc_ny_test_hourly.csv", index_col= "UTC_Timestamp")

train_data.columns = ["Actual_Load_MW", "Temperature_Fahrenheit", "LMP", "Congestion"]
test_data.columns = ["Actual_Load_MW", "Temperature_Fahrenheit", "LMP", "Congestion"]
# -

# Fill the missing test data with median
test_data = test_data.apply(lambda col: col.fillna(col.median()), axis = 0)

minMaxScaler = MinMaxScaler(feature_range=(0,1))
scaled_train_data = minMaxScaler.fit_transform(train_data)
scaled_test_data = minMaxScaler.transform(test_data)


# ## Filter anomalous training data

def objective(trial: optuna.Trial, X_train: pd.DataFrame) -> float:
    n_estimators = trial.suggest_int("n_estimators", 50, 500, step=50)
    max_samples = trial.suggest_int("max_samples", 32, 256, step=32)
    max_features = trial.suggest_float("max_features", 0.5, 1.0, step=0.1)
    contamination = trial.suggest_float("contamination", 0.01, 0.1, step=0.01)
    bootstrap = False
    random_state = 13

    iso_forest = IsolationForest(
        n_estimators = n_estimators,
        max_samples = max_samples,
        max_features = max_features,
        contamination = contamination,
        bootstrap = bootstrap,
        random_state = random_state
    )

    iso_forest.fit(X_train)
    anomaly_scores = iso_forest.decision_function(X_train)
    return np.var(anomaly_scores) # optimize on variance => higher variance shows better separation between non-anomalous and anomalous data


# +
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=13))
study.optimize(lambda trial: objective(trial, scaled_train_data), n_trials=30)

print("Best hyperparameters:", study.best_params)

# +
iso_forest = IsolationForest(**study.best_params)
iso_forest.fit(scaled_train_data)

# Compute anomaly scores and calculate threshold
anomaly_scores = iso_forest.decision_function(scaled_train_data)
extreme_threshold = np.percentile(anomaly_scores, 5)

# Keep only normal data
extreme_anomalies = scaled_train_data[anomaly_scores < extreme_threshold]
X_filtered = scaled_train_data[anomaly_scores >= extreme_threshold]

# Display results
print(f"Number of extreme anomalies detected: {len(extreme_anomalies)}")
print(f"Remaining data after filtering: {X_filtered.shape}")


# -

# ## Autoencoder Training

# Create sequences for multivariate time-series
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)


SEQ_LENGTH = 24  # 24 timesteps
BATCH_SIZE = 32
EPOCHS = 30

X_train = create_sequences(X_filtered, SEQ_LENGTH)
X_test = create_sequences(scaled_test_data, SEQ_LENGTH)

input_dim = X_train.shape[2] # number of features
input_layer = Input(shape=(SEQ_LENGTH, input_dim), name = 'input_layer')

# +
# Encoder
encoder_lstm_1 = LSTM(64, return_sequences=True, name="encoder_lstm_1")(input_layer)
encoder_lstm_2 = LSTM(32, return_sequences=True, name="encoder_lstm_2")(encoder_lstm_1)

# Attention mechanism
attention_layer = Attention(use_scale =True, name="attention_layer")([encoder_lstm_2, encoder_lstm_2]) # TODO: review

# Decoder
decoder_lstm_1 = LSTM(32, return_sequences=True, name="decoder_lstm_1")(attention_layer)
decoder_lstm_2 = LSTM(64, return_sequences=True, name="decoder_lstm_2")(decoder_lstm_1)
decoder_output = TimeDistributed(Dense(4), name="decoder_output")(decoder_lstm_2)

# Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoder_output, name="autoencoder")

# +
optimizer = Adam(learning_rate=0.0001, clipnorm=1)
autoencoder.compile(optimizer=optimizer, loss='mse')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
# -

# Train the model
history = autoencoder.fit(
    X_train, X_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, lr_scheduler]
)

# Extract the loss and validation loss from the history
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Training Loss', color='blue')
plt.plot(val_loss, label='Validation Loss', color='orange')
plt.title('Loss Convergence During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Calculate reconstruction error for each instance from test data
test_reconstructed = autoencoder.predict(X_test)
reconstruction_error = np.mean(np.square(X_test - test_reconstructed), axis=(1, 2))  # MSE per sample
print(reconstruction_error)

# Calculate reconstruction error for training or validation data
train_reconstructed = autoencoder.predict(X_train)
train_error = np.mean(np.square(X_train - train_reconstructed), axis=(1, 2))
print(train_error)

# Set threshold as the 95th percentile of training reconstruction error
threshold = np.percentile(train_error, 95)
print(f"Anomaly threshold: {threshold}")

sns.histplot(train_error, bins=50, kde=True)
plt.axvline(threshold, color='red', linestyle='dashed', label="Threshold (90%)")
plt.legend()
plt.title("Reconstruction Train Error Distribution")
plt.show()

# Most values are clustered near zero and so autoencoder is fine.

# Detect anomalies in both traina and test data and label anomaly as 1
anomaly_labels_test = (reconstruction_error > threshold).astype(int)
anomaly_labels_train = (train_error > threshold).astype(int)

print(f"Number of test anomalies detected: {np.sum(anomaly_labels_test)}")

# +
# Show indices of anomalies in test data
anomaly_indices = np.where(anomaly_labels_test)[0]

# Plot reconstruction error
plt.figure(figsize=(12, 6))
plt.plot(reconstruction_error, label='Test Reconstruction Error')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.scatter(anomaly_indices, reconstruction_error[anomaly_indices], color='red', label='Anomalies')
plt.xlabel('Test Sample Index')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Error and Anomaly Detection')
plt.legend()
plt.tight_layout()
plt.show()
# -

# ## XGBoost Anomaly Classification

# +
# Flatten time-series features for XGBoost input (samples, timesteps*features)
X_train_flat = pd.DataFrame(X_train.reshape(len(X_train), -1))
X_test_flat = pd.DataFrame(X_test.reshape(len(X_test), -1))

# Append anomaly labels (1 and 0) under new column 'anomaly label'
X_train_flat['anomaly_label'] = anomaly_labels_train
X_test_flat['anomaly_label'] = anomaly_labels_test

# +
# Separate features (X) and labels (y)
X_train_xgb = X_train_flat.drop(columns=['anomaly_label'])
y_train_xgb = X_train_flat['anomaly_label']

X_test_xgb = X_test_flat.drop(columns=['anomaly_label'])
y_test_xgb = X_test_flat['anomaly_label']

# +
# Average across time steps for each sample
X_train_xgb = X_train.mean(axis=1)  # Shape: (num_samples, 4)
X_test_xgb = X_test.mean(axis=1)

print("X_train_xgb shape:", X_train_xgb.shape)
print("X_test_xgb shape:", X_test_xgb.shape)
# -

# Apply SMOTE to balance the classes
smote_tomek = SMOTETomek(sampling_strategy = 0.5, random_state=13)
X_train_smote, y_train_smote = smote_tomek.fit_resample(X_train_xgb, y_train_xgb)

# +
param_dist = {
    'n_estimators': np.arange(200, 500, 50),
    'max_depth': np.arange(4, 9, 1),  # Too deep trees can lead to overfitting
    'min_child_weight': np.arange(1, 5, 1),  # Higher values reduce overfitting
    'learning_rate': np.linspace(0.01, 0.3, 10),  # Avoid very small or large learning rates
    'subsample': np.linspace(0.5, 1.0, 5),
    'colsample_bytree': np.linspace(0.5, 1.0, 5),
    'lambda': np.arange(0, 4, 1),  # L2 regularization, commonly 1-3
    'alpha': np.arange(0, 4, 1)  # L1 regularization, commonly 0-3
}

# Create the XGBoost model
model_xgb = xgb.XGBClassifier(random_state=13)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    model_xgb,
    param_distributions=param_dist,
    cv=5,
    scoring="recall",
)

random_search.fit(X_train_smote, y_train_smote)
print("Best Parameters:", random_search.best_params_)
# -

xgb_best = random_search.best_estimator_
y_pred_xgb = xgb_best.predict(X_test_xgb)
print(classification_report(y_test_xgb, y_pred_xgb, target_names=['Normal', 'Anomaly']))

# ## SHAP Plots

shap.initjs()

# Initialize SHAP explainer for XGBoost
explainer = shap.TreeExplainer(xgb_best)
shap_values = explainer(X_test_xgb)

shap.summary_plot(shap_values, X_test_xgb, feature_names=['Load', 'TempF', 'LMP', 'Congestion'], rng=13)

feature_names = ["load_MW", "Temp_Fahrenheit", "LMP", "Congestion"]
X_test_xgb_df = pd.DataFrame(X_test_xgb, columns=feature_names)
X_test_xgb_df.head()

# +
# # Compute SHAP values
explainer = shap.TreeExplainer(xgb_best)  # Ensure the model is trained on the same features
shap_values = explainer(X_test_xgb_df)

# Scale back X_test_xgb to original values
X_test_xgb_original = minMaxScaler.inverse_transform(X_test_xgb_df)
shap_values.data = X_test_xgb_original  # Ensuring SHAP reflects true original feature values

# Waterfall plot for the first test sample
shap.plots.waterfall(shap_values[15])
# -

# visualize the first prediction's explanation with a force plot
shap.plots.force(shap_values[15])

# create a dependence scatter plot to show the effect of a single feature across the whole dataset
shap.plots.scatter(shap_values[:, 2], color=shap_values[:, 0])

# create a dependence scatter plot to show the effect of a single feature across the whole dataset
shap.plots.scatter(shap_values[:, 0], color=shap_values[:, 1])

# create a dependence scatter plot to show the effect of a single feature across the whole dataset
shap.plots.scatter(shap_values[:, 2], color=shap_values[:, 3])

shap.plots.bar(shap_values)


# +
# shap.plots.heatmap(shap_values) # TODO: running for 1+ hr but did not finish
