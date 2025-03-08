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

# # NYC: Traditional Modeling

# ## Libraries

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import xgboost as xgb
import pmdarima as pm
import optuna

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from classes.helper_functions import is_stationary
# -

random.seed(13)
np.random.seed(13)

# ## Import Data

# +
nyc_train = pd.read_csv("../data/nyc_ny_train_hourly_interpolated.csv", index_col= "UTC_Timestamp")
nyc_test = pd.read_csv("../data/nyc_ny_test_hourly.csv", index_col= "UTC_Timestamp")

nyc_train.columns = ["Actual_Load_MW", "Temperature_Fahrenheit", "LMP", "Congestion"]
nyc_test.columns = ["Actual_Load_MW", "Temperature_Fahrenheit", "LMP", "Congestion"]
# -

nyc_train.info()
nyc_train.head()

nyc_test.info()
nyc_test.head()

# ## SARIMA Modeling

# +
nyc_train = pd.read_csv("../data/nyc_ny_train_hourly_interpolated.csv", index_col= "UTC_Timestamp")
nyc_test = pd.read_csv("../data/nyc_ny_test_hourly.csv", index_col= "UTC_Timestamp")

nyc_train.columns = ["Actual_Load_MW", "Temperature_Fahrenheit", "LMP", "Congestion"]
nyc_test.columns = ["Actual_Load_MW", "Temperature_Fahrenheit", "LMP", "Congestion"]

nyc_train.index = pd.to_datetime(nyc_train.index)
nyc_test.index = pd.to_datetime(nyc_test.index)
# -

# ### Check Stationarity and plot ACF/PACF

print(is_stationary(nyc_train["Actual_Load_MW"], significance=0.05))
print("\n")
print(is_stationary(nyc_train["Temperature_Fahrenheit"], significance=0.05))
print("\n")
print(is_stationary(nyc_train["LMP"], significance=0.05))
print("\n")
print(is_stationary(nyc_train["Congestion"], significance=0.05))

plot_acf(nyc_train["Actual_Load_MW"], alpha=0.05).show()
plot_pacf(nyc_train["Actual_Load_MW"], alpha=0.05).show()

plot_acf(nyc_train["Temperature_Fahrenheit"], alpha=0.05).show()
plot_pacf(nyc_train["Temperature_Fahrenheit"], alpha=0.05).show()

plot_acf(nyc_train["LMP"], alpha=0.05).show()
plot_pacf(nyc_train["LMP"], alpha=0.05).show()

plot_acf(nyc_train["Congestion"], alpha=0.05).show()
plot_pacf(nyc_train["Congestion"], alpha=0.05).show()

# ### Hyperparameter Tuning

# +
"""
Auto ARIMA requires prohibitive amount of time and resources for hourly dataset of this size.
Attempting a daily aggregation instead.
"""

nyc_train_daily = nyc_train.resample("D").agg({
    "Actual_Load_MW": "sum",
    "Temperature_Fahrenheit": "mean",
    "LMP": "mean",
    "Congestion": "mean"
})
nyc_test_daily = nyc_test.resample("D").agg({
    "Actual_Load_MW": "sum",
    "Temperature_Fahrenheit": "mean",
    "LMP": "mean",
    "Congestion": "mean"
})

# -

nyc_train_daily.describe()

nyc_test_daily.describe()

nyc_test_daily.interpolate(method="time", inplace=True)

sarima_fit = pm.auto_arima(y=nyc_train_daily["Actual_Load_MW"],
                           X=nyc_train_daily[["Temperature_Fahrenheit", "LMP", "Congestion"]],
                           stationary=True,
                           start_p=1, start_q=1, 
                           max_p=15, max_q=15,
                           start_P=1, start_Q=1,
                           max_P=15, max_Q=15,
                           m = 12,
                           seasonal=True,
                           suppress_warnings=True,
                           trace=True,
                           stepwise=True)
print(sarima_fit.summary())

# ### SARIMA Inference

sarima_pred = sarima_fit.predict(n_periods=len(nyc_test_daily),
                                 X=nyc_test_daily[["Temperature_Fahrenheit", "LMP", "Congestion"]])
sarima_pred.head()

sarima_rmse = root_mean_squared_error(y_true=nyc_test_daily["Actual_Load_MW"],
                                       y_pred=sarima_pred)
print(sarima_rmse)

# Sarima fails to perform effectively with data that has long-range dependencies, such as in this dataset. Therefore, switching to XGBoost as the traditional model baseline for the overall analysis. 

# ## XGBoost Modeling

# +
nyc_train_xgb = pd.read_csv("../data/nyc_ny_train_hourly_interpolated.csv", index_col= "UTC_Timestamp")
nyc_test_xgb = pd.read_csv("../data/nyc_ny_test_hourly.csv", index_col= "UTC_Timestamp")

nyc_train_xgb.columns = ["Actual_Load_MW", "Temperature_Fahrenheit", "LMP", "Congestion"]
nyc_test_xgb.columns = ["Actual_Load_MW", "Temperature_Fahrenheit", "LMP", "Congestion"]
# -

# ### Feature Engineering

sequence = 24
nyc_test_xgb.ffill(inplace=True)

# +
xgb_X_train = nyc_train_xgb.copy()
xgb_y_train = nyc_train_xgb[["Actual_Load_MW"]]

for s in range(1, sequence + 1):
    xgb_X_train[f"load_lag_{s}"] = xgb_X_train[["Actual_Load_MW"]].shift(s)
    xgb_X_train[f"temp_lag_{s}"] = xgb_X_train[["Temperature_Fahrenheit"]].shift(s)
    xgb_X_train[f"lmp_lag_{s}"] = xgb_X_train[["LMP"]].shift(s)
    xgb_X_train[f"congestion_lag_{s}"] = xgb_X_train[["Congestion"]].shift(s)

xgb_X_train = xgb_X_train.drop(["Actual_Load_MW", "Temperature_Fahrenheit", "LMP", "Congestion"], axis = 1)
xgb_X_train.dropna(axis = 0, inplace = True)
xgb_y_train = xgb_y_train[24:]

# +
xgb_X_test = nyc_test_xgb.copy()
xgb_y_test = nyc_test_xgb[["Actual_Load_MW"]]

for s in range(1, sequence + 1):
    xgb_X_test[f"load_lag_{s}"] = xgb_X_test[["Actual_Load_MW"]].shift(s)
    xgb_X_test[f"temp_lag_{s}"] = xgb_X_test[["Temperature_Fahrenheit"]].shift(s)
    xgb_X_test[f"lmp_lag_{s}"] = xgb_X_test[["LMP"]].shift(s)
    xgb_X_test[f"congestion_lag_{s}"] = xgb_X_test[["Congestion"]].shift(s)

xgb_X_test = xgb_X_test.drop(["Actual_Load_MW", "Temperature_Fahrenheit", "LMP", "Congestion"], axis = 1)
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

# Tune using Bayesian Optimization with Optuna

def objective(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.DataFrame) -> float:
    params = {
        "objective": "reg:squarederror",
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": 13
    }

    xgb_model = xgb.XGBRegressor(**params)
    score = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring="neg_mean_squared_error").mean()
    return score


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=13))
study.optimize(lambda trial: objective(trial, xgb_X_train, xgb_y_train), n_trials=15)
best_params = study.best_params
print(best_params)

xgb_model = xgb.XGBRegressor(**best_params)
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
