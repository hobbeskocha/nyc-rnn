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

# # NYC: EDA

# ## Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.max_columns = 40
pd.options.display.max_rows = 100

# ## Import Data

nyc_load_2021 = pd.read_csv("../data/nyiso_load_act_hr_2021.csv", header = 3)
nyc_load_2022 = pd.read_csv("../data/nyiso_load_act_hr_2022.csv", header = 3)
nyc_load_2023 = pd.read_csv("../data/nyiso_load_act_hr_2023.csv", header = 3)
nyc_load_2024 = pd.read_csv("../data/nyiso_load_act_hr_2024.csv", header = 3)

nyc_temp_2021 = pd.read_csv("../data/nyiso_temp_hr_2021.csv", header = 3)
nyc_temp_2022 = pd.read_csv("../data/nyiso_temp_hr_2022.csv", header = 3)
nyc_temp_2023 = pd.read_csv("../data/nyiso_temp_hr_2023.csv", header = 3)
nyc_temp_2024 = pd.read_csv("../data/nyiso_temp_hr_2024.csv", header = 3)

nyc_price_2021 = pd.read_csv("../data/nyiso_price_zones_2021.csv", header = 3)
nyc_price_2022 = pd.read_csv("../data/nyiso_price_zones_2022.csv", header = 3)
nyc_price_2023 = pd.read_csv("../data/nyiso_price_zones_2023.csv", header = 3)
nyc_price_2024 = pd.read_csv("../data/nyiso_price_zones_2024.csv", header = 3)

# ## Define Dataframes

print(list(nyc_load_2021.columns) == list(nyc_load_2022.columns) 
      and list(nyc_load_2022.columns) == list(nyc_load_2023.columns) 
      and list(nyc_load_2023.columns) == list(nyc_load_2024.columns))

print(list(nyc_temp_2021.columns) == list(nyc_temp_2022.columns) 
      and list(nyc_temp_2022.columns) == list(nyc_temp_2023.columns) 
      and list(nyc_temp_2023.columns) == list(nyc_temp_2024.columns))

print(list(nyc_price_2021.columns) == list(nyc_price_2022.columns)
      and list(nyc_price_2022.columns) == list(nyc_price_2023.columns)
      and list(nyc_price_2023.columns) == list(nyc_price_2024.columns))

# +
nyc_load_complete = pd.concat([nyc_load_2021, nyc_load_2022, nyc_load_2023, nyc_load_2024], axis = 0)
print(nyc_load_complete.shape)

nyc_load_complete.head()

# +
nyc_temperature_complete = pd.concat([nyc_temp_2021, nyc_temp_2022, nyc_temp_2023, nyc_temp_2024], axis = 0)
print(nyc_temperature_complete.shape)

nyc_temperature_complete.head()

# +
nyc_price_complete = pd.concat([nyc_price_2021, nyc_price_2022, nyc_price_2023, nyc_price_2024], axis = 0)
print(nyc_price_complete.shape)

nyc_price_complete.head()
# -

nyc_load_complete.to_csv("../data/nyc_complete.csv", index = False)
nyc_temperature_complete.to_csv("../data/nyc_temp_complete.csv", index=False)
nyc_price_complete.to_csv("../data/nyc_price_complete.csv", index=False)

# ## Data Processing and Exploratory Data Analysis

nyc_load_complete.info()

nyc_temperature_complete.info()

nyc_price_complete.info()

nyc_load_complete.isna().sum().sort_values()

nyc_temperature_complete.isna().sum().sort_values()

nyc_price_complete.isna().sum().sort_values()

nyc_load_complete.columns = nyc_load_complete.columns.str.replace(r'[ \(\)\-]', '_', regex = True)
nyc_load_complete.columns

nyc_temperature_complete.columns = nyc_temperature_complete.columns.str.replace(r'[ \(\)\-\/]', '_', regex = True)
nyc_temperature_complete.columns

nyc_price_complete.columns = nyc_price_complete.columns.str.replace(r'[ \(\)\-]', '_', regex = True)
nyc_price_complete.columns

nyc_load_complete["UTC_Timestamp__Interval_Ending_"] = pd.to_datetime(nyc_load_complete["UTC_Timestamp__Interval_Ending_"], format = "%Y-%m-%d %H:%M:%S")
nyc_temperature_complete["UTC_Timestamp__Interval_Ending_"] = pd.to_datetime(nyc_temperature_complete["UTC_Timestamp__Interval_Ending_"], format = "%Y-%m-%d %H:%M:%S")
nyc_price_complete["UTC_Timestamp__Interval_Ending_"] = pd.to_datetime(nyc_price_complete["UTC_Timestamp__Interval_Ending_"], format = "%Y-%m-%d %H:%M")

nyc_ny_load = nyc_load_complete[["UTC_Timestamp__Interval_Ending_", "J___New_York_City_Actual_Load__MW_"]]
nyc_ny_load.columns = ["UTC_Timestamp__Interval_Ending_", "New_York_City_Actual_Load__MW_"]
nyc_ny_load.describe()

nyc_ny_temperature = nyc_temperature_complete[["UTC_Timestamp__Interval_Ending_", "New_York_City___JFK_Airport_Temperature__Fahrenheit_"]]
nyc_ny_temperature.columns = ["UTC_Timestamp__Interval_Ending_", "New_York_City___JFK_Airport_Temperature__Fahrenheit_"]
nyc_ny_temperature.describe()

nyc_ny_price = nyc_price_complete[["UTC_Timestamp__Interval_Ending_",
                                    "J___New_York_City_LMP", 
                                    "J___New_York_City__Congestion_"]]
nyc_ny_price.columns = ["UTC_Timestamp__Interval_Ending_", "New_York_City_LMP", "New_York_City__Congestion_"]
nyc_ny_price.describe()

# ### Include Full Date Range

# +
datetime_index = pd.DataFrame(pd.date_range(start = nyc_ny_load["UTC_Timestamp__Interval_Ending_"].min(),
                               end = nyc_ny_load["UTC_Timestamp__Interval_Ending_"].max(),
                               freq = "1h"))
datetime_index.columns = ["UTC_Timestamp"]

temp_nyc_load = datetime_index.merge(nyc_ny_load, how = "left",
                                 left_on = "UTC_Timestamp", right_on = "UTC_Timestamp__Interval_Ending_")

temp_nyc_load.index = temp_nyc_load["UTC_Timestamp"]
temp_nyc_load.drop(["UTC_Timestamp", "UTC_Timestamp__Interval_Ending_"], axis = 1, inplace= True)
nyc_ny_load = temp_nyc_load

# +
datetime_index = pd.DataFrame(pd.date_range(start = nyc_ny_temperature["UTC_Timestamp__Interval_Ending_"].min(),
                               end = nyc_ny_temperature["UTC_Timestamp__Interval_Ending_"].max(),
                               freq = "1h"))
datetime_index.columns = ["UTC_Timestamp"]

temp_nyc_temp = datetime_index.merge(nyc_ny_temperature, how = "left",
                                      left_on = "UTC_Timestamp", right_on = "UTC_Timestamp__Interval_Ending_")

temp_nyc_temp.index = temp_nyc_temp["UTC_Timestamp"]
temp_nyc_temp.drop(["UTC_Timestamp", "UTC_Timestamp__Interval_Ending_"], axis = 1, inplace= True)
nyc_ny_temperature = temp_nyc_temp

# +
datetime_index = pd.DataFrame(pd.date_range(start = nyc_ny_price["UTC_Timestamp__Interval_Ending_"].min(),
                               end = nyc_ny_price["UTC_Timestamp__Interval_Ending_"].max(),
                               freq = "1h"))
datetime_index.columns = ["UTC_Timestamp"]

temp_nyc_price = datetime_index.merge(nyc_ny_price, how = "left",
                                       left_on = "UTC_Timestamp", right_on = "UTC_Timestamp__Interval_Ending_")

temp_nyc_price.index = temp_nyc_price["UTC_Timestamp"]
temp_nyc_price.drop(["UTC_Timestamp", "UTC_Timestamp__Interval_Ending_"], axis = 1, inplace= True)
nyc_ny_price = temp_nyc_price
# -

nyc_ny_load = pd.merge(nyc_ny_load, nyc_ny_temperature, on="UTC_Timestamp", how="left")
nyc_ny_load = pd.merge(nyc_ny_load, nyc_ny_price, on="UTC_Timestamp", how="left")
print(nyc_ny_load.shape)
nyc_ny_load.info()

# ### Define Train-Test Split

# +
nyc_ny_load.reset_index(inplace=True)
nyc_ny_load = nyc_ny_load.sort_values(by = "UTC_Timestamp")

nyc_ny_train = nyc_ny_load[nyc_ny_load["UTC_Timestamp"].dt.year < 2024]
nyc_ny_test = nyc_ny_load[nyc_ny_load["UTC_Timestamp"].dt.year >= 2024]

print(nyc_ny_train.shape, nyc_ny_test.shape)
# -

nyc_ny_train.describe()

nyc_ny_test.describe()


# ### Impute Missing Values for Train Set

# +
# def impute_missing_by_seasonal_average(df):
#     for i in range(len(df)):
#         if pd.isna(df.loc[df.index[i], "New_York_City_Actual_Load__MW_"]):
#             # Find all previous similar periods (same month, day, and hour) for load
#             similar_periods = df[(df['month'] == df['month'].iloc[i]) &
#                                  (df['day'] == df['day'].iloc[i]) &
#                                  (df['hour'] == df['hour'].iloc[i]) &
#                                  (df.index < df.index[i])]
#             if not similar_periods.empty:
#                 df.loc[df.index[i], "New_York_City_Actual_Load__MW_"] = similar_periods["New_York_City_Actual_Load__MW_"].mean()
        
#         if pd.isna(df.loc[df.index[i], "New_York_City___JFK_Airport_Temperature__Fahrenheit_"]):
#             # Find all previous similar periods (same month, day, and hour) for temperature
#             similar_periods = df[(df['month'] == df['month'].iloc[i]) &
#                                  (df['day'] == df['day'].iloc[i]) &
#                                  (df['hour'] == df['hour'].iloc[i]) &
#                                  (df.index < df.index[i])]
#             if not similar_periods.empty:
#                 df.loc[df.index[i], "New_York_City___JFK_Airport_Temperature__Fahrenheit_"] = similar_periods["New_York_City___JFK_Airport_Temperature__Fahrenheit_"].mean()
        
#         if pd.isna(df.loc[df.index[i], "New_York_City_LMP"]):
#             # Find all previous similar periods (same month, day, and hour) for LMP
#             similar_periods = df[(df['month'] == df['month'].iloc[i]) &
#                                  (df['day'] == df['day'].iloc[i]) &
#                                  (df['hour'] == df['hour'].iloc[i]) &
#                                  (df.index < df.index[i])]
#             if not similar_periods.empty:
#                 df.loc[df.index[i], "New_York_City_LMP"] = similar_periods["New_York_City_LMP"].mean()

#     return df
# -

def impute_missing_by_seasonal_average(df: pd.DataFrame) -> pd.DataFrame:
    group_by_cols = ["month", "day", "hour"]
    load_mean = df.groupby(group_by_cols)["New_York_City_Actual_Load__MW_"].transform(lambda x: x.ffill().mean())
    temp_mean = df.groupby(group_by_cols)["New_York_City___JFK_Airport_Temperature__Fahrenheit_"].transform(lambda x: x.ffill().mean())
    lmp_mean = df.groupby(group_by_cols)["New_York_City_LMP"].transform(lambda x: x.ffill().mean())
    congestion_mean = df.groupby(group_by_cols)["New_York_City__Congestion_"].transform(lambda x: x.ffill().mean())

    df["New_York_City_Actual_Load__MW_"] = df["New_York_City_Actual_Load__MW_"].fillna(load_mean)
    df["New_York_City___JFK_Airport_Temperature__Fahrenheit_"] = df["New_York_City___JFK_Airport_Temperature__Fahrenheit_"].fillna(temp_mean)
    df["New_York_City_LMP"] = df["New_York_City_LMP"].fillna(lmp_mean)
    df["New_York_City__Congestion_"] = df["New_York_City__Congestion_"].fillna(congestion_mean)

    return df


# +
nyc_ny_train.index = nyc_ny_train["UTC_Timestamp"]
nyc_ny_train = nyc_ny_train.drop(["UTC_Timestamp"], axis=1)
temp = nyc_ny_train

temp["month"] = temp.index.month
temp["day"] = temp.index.day
temp["hour"] = temp.index.hour

temp_interpolate = impute_missing_by_seasonal_average(temp)
temp_interpolate = temp_interpolate.interpolate(method = "time")

temp_interpolate = temp_interpolate.drop(columns=["month", "day", "hour"])

print(temp_interpolate.info())
temp_interpolate.head()

# +
nyc_ny_train = temp_interpolate
nyc_ny_train_daily = nyc_ny_train.resample("D").agg({
    "New_York_City_Actual_Load__MW_": "sum",
    "New_York_City___JFK_Airport_Temperature__Fahrenheit_": "mean",
    "New_York_City_LMP": "mean",
    "New_York_City__Congestion_": "mean"
})

nyc_ny_train.to_csv("../data/nyc_ny_train_hourly_interpolated.csv")
# -

print(nyc_ny_train_daily.shape)
nyc_ny_train_daily.head()

nyc_ny_test.index = nyc_ny_test["UTC_Timestamp"]
nyc_ny_test = nyc_ny_test.drop(["UTC_Timestamp"], axis = 1)
nyc_ny_test_daily = nyc_ny_test.resample("D").agg({
    "New_York_City_Actual_Load__MW_": "sum",
    "New_York_City___JFK_Airport_Temperature__Fahrenheit_": "mean",
    "New_York_City_LMP": "mean",
    "New_York_City__Congestion_": "mean"
})
nyc_ny_test.to_csv("../data/nyc_ny_test_hourly.csv")

print(nyc_ny_test_daily.shape)
nyc_ny_test_daily.head()

# ### Data Visualization

nyc_viz = nyc_ny_train.copy()
nyc_viz["year"] = nyc_viz.index.year
nyc_viz["quarter"] = nyc_viz.index.quarter
nyc_viz["month"] = nyc_viz.index.month
nyc_viz["hour"] = nyc_viz.index.hour

# +
nyc_load_hour = sns.boxplot(nyc_viz, x = "hour", y = "New_York_City_Actual_Load__MW_").set(title = "NYC Load by Hour")
plt.savefig("../artifacts/nyc-box-hour.png")
plt.show()

nyc_load_month = sns.boxplot(nyc_viz, x = "month", y = "New_York_City_Actual_Load__MW_").set(title = "NYC Load by Month")
plt.savefig("../artifacts/nyc-box-month.png")
plt.show()

nyc_load_quarter = sns.boxplot(nyc_viz, x = "quarter", y = "New_York_City_Actual_Load__MW_").set(title = "NYC Load by Quarter")
plt.savefig("../artifacts/nyc-box-quarter.png")
plt.show()

nyc_load_year = sns.boxplot(nyc_viz, x = "year", y = "New_York_City_Actual_Load__MW_").set(title = "NYC Load by Year")
plt.savefig("../artifacts/nyc-box-year.png")
plt.show()

# +
sns.boxplot(nyc_viz, x = "hour", y = "New_York_City___JFK_Airport_Temperature__Fahrenheit_").set(title = "NYC Temperature by Hour")
plt.show()

sns.boxplot(nyc_viz, x = "month", y = "New_York_City___JFK_Airport_Temperature__Fahrenheit_").set(title = "NYC Temperature by Month")
plt.show()

sns.boxplot(nyc_viz, x = "quarter", y = "New_York_City___JFK_Airport_Temperature__Fahrenheit_").set(title = "NYC Temperature by Quarter")
plt.show()

sns.boxplot(nyc_viz, x = "year", y = "New_York_City___JFK_Airport_Temperature__Fahrenheit_").set(title = "NYC Temperature by Year")
plt.show()

# +
sns.boxplot(nyc_viz, x = "hour", y = "New_York_City_LMP").set(title = "NYC LMP by Hour")
plt.show()

sns.boxplot(nyc_viz, x = "month", y = "New_York_City_LMP").set(title = "NYC LMP by Month")
plt.show()

sns.boxplot(nyc_viz, x = "quarter", y = "New_York_City_LMP").set(title = "NYC LMP by Quarter")
plt.show()

sns.boxplot(nyc_viz, x = "year", y = "New_York_City_LMP").set(title = "NYC LMP by Year")
plt.show()

# +
sns.boxplot(nyc_viz, x = "hour", y = "New_York_City__Congestion_").set(title = "NYC Congestion by Hour")
plt.show()

sns.boxplot(nyc_viz, x = "month", y = "New_York_City__Congestion_").set(title = "NYC Congestion by Month")
plt.show()

sns.boxplot(nyc_viz, x = "quarter", y = "New_York_City__Congestion_").set(title = "NYC Congestion by Quarter")
plt.show()

sns.boxplot(nyc_viz, x = "year", y = "New_York_City__Congestion_").set(title = "NYC Congestion by Year")
plt.show()

# +
sns.histplot(nyc_ny_train_daily["New_York_City_Actual_Load__MW_"]).set(title = "NYC Load Daily")
plt.savefig("../artifacts/nyc-hist-load-daily.png")
plt.show()

sns.histplot(nyc_ny_train["New_York_City_Actual_Load__MW_"]).set(title = "NYC Load Hourly")
plt.savefig("../artifacts/nyc-hist-load-hourly.png")
plt.show()

# +
sns.histplot(nyc_ny_train_daily["New_York_City___JFK_Airport_Temperature__Fahrenheit_"]).set(title = "NYC Temperature Daily")
plt.show()

sns.histplot(nyc_ny_train["New_York_City___JFK_Airport_Temperature__Fahrenheit_"]).set(title = "NYC Temperature Hourly")
plt.show()

# +
sns.histplot(nyc_ny_train_daily["New_York_City_LMP"]).set(title = "NYC LMP Daily")
plt.show()

sns.histplot(nyc_ny_train["New_York_City_LMP"]).set(title = "NYC LMP Hourly")
plt.show()

# +
sns.histplot(nyc_ny_train_daily["New_York_City__Congestion_"]).set(title = "NYC Congestion Daily")
plt.show()

sns.histplot(nyc_ny_train["New_York_City__Congestion_"]).set(title = "NYC Congestion Hourly")
plt.show()

# +
plt.figure(figsize = (18, 8))
sns.lineplot(data = nyc_ny_train_daily, x = nyc_ny_train_daily.index, y = "New_York_City_Actual_Load__MW_").set(title = "NYC Daily Load for Training Set")
plt.xticks(rotation = 45)
plt.savefig("../artifacts/nyc-training-load.png")
plt.show()

plt.figure(figsize = (18, 8))
sns.lineplot(data = nyc_ny_train_daily, x = nyc_ny_train_daily.index, y = "New_York_City___JFK_Airport_Temperature__Fahrenheit_").set(title = "NYC Daily Temperature for Training Set")
plt.xticks(rotation = 45)
plt.savefig("../artifacts/nyc-training-temperature.png")
plt.show()

plt.figure(figsize = (18, 8))
sns.lineplot(data = nyc_ny_train_daily, x = nyc_ny_train_daily.index, y = "New_York_City_LMP").set(title = "NYC Daily LMP for Training Set")
plt.xticks(rotation = 45)
plt.savefig("../artifacts/nyc-training-lmp.png")
plt.show()

plt.figure(figsize = (18, 8))
sns.lineplot(data = nyc_ny_train_daily, x = nyc_ny_train_daily.index, y = "New_York_City__Congestion_").set(title = "NYC Daily Congestion for Training Set")
plt.xticks(rotation = 45)
plt.savefig("../artifacts/nyc-training-congestion.png")
plt.show()

# +
plt.figure(figsize = (18, 8))
sns.lineplot(data = nyc_ny_test_daily, x = nyc_ny_test_daily.index, y = "New_York_City_Actual_Load__MW_").set(title = "NYC Daily Load for Test Set")
plt.xticks(rotation = 45)
plt.show()

plt.figure(figsize = (18, 8))
sns.lineplot(data = nyc_ny_test_daily, x = nyc_ny_test_daily.index, y = "New_York_City___JFK_Airport_Temperature__Fahrenheit_").set(title = "NYC Daily Temperature for Test Set")
plt.xticks(rotation = 45)
plt.show()

plt.figure(figsize = (18, 8))
sns.lineplot(data = nyc_ny_test_daily, x = nyc_ny_test_daily.index, y = "New_York_City_LMP").set(title = "NYC Daily LMP for Test Set")
plt.xticks(rotation = 45)
plt.show()

plt.figure(figsize = (18, 8))
sns.lineplot(data = nyc_ny_test_daily, x = nyc_ny_test_daily.index, y = "New_York_City__Congestion_").set(title = "NYC Daily Congestion for Test Set")
plt.xticks(rotation = 45)
plt.show()
