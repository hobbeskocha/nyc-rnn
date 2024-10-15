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

nyc_2021 = pd.read_csv("../data/nyiso_load_act_hr_2021.csv", header = 3)
nyc_2022 = pd.read_csv("../data/nyiso_load_act_hr_2022.csv", header = 3)
nyc_2023 = pd.read_csv("../data/nyiso_load_act_hr_2023.csv", header = 3)
nyc_2024 = pd.read_csv("../data/nyiso_load_act_hr_2024.csv", header = 3)

nyc_temp_2021 = pd.read_csv("../data/nyiso_temp_hr_2021.csv", header = 3)
nyc_temp_2022 = pd.read_csv("../data/nyiso_temp_hr_2022.csv", header = 3)
nyc_temp_2023 = pd.read_csv("../data/nyiso_temp_hr_2023.csv", header = 3)
nyc_temp_2024 = pd.read_csv("../data/nyiso_temp_hr_2024.csv", header = 3)

# ## Define Dataframes

print(list(nyc_2021.columns) == list(nyc_2022.columns) 
      and list(nyc_2022.columns) == list(nyc_2023.columns) 
      and list(nyc_2023.columns) == list(nyc_2024.columns))

print(list(nyc_temp_2021.columns) == list(nyc_temp_2022.columns) 
      and list(nyc_temp_2022.columns) == list(nyc_temp_2023.columns) 
      and list(nyc_temp_2023.columns) == list(nyc_temp_2024.columns))

# +
nyc_complete = pd.concat([nyc_2021, nyc_2022, nyc_2023, nyc_2024], axis = 0)
print(nyc_complete.shape)

nyc_complete.head()

# +
nyc_temperature_complete = pd.concat([nyc_temp_2021, nyc_temp_2022, nyc_temp_2023, nyc_temp_2024], axis = 0)
print(nyc_temperature_complete.shape)

nyc_temperature_complete.head()
# -

nyc_complete.to_csv("../data/nyc_complete.csv", index = False)
nyc_temperature_complete.to_csv("../data/nyc_temp_complete.csv", index=False)

# ## Data Processing and Exploratory Data Analysis

nyc_complete.info()

nyc_temperature_complete.info()

nyc_complete.isna().sum().sort_values()

nyc_temperature_complete.isna().sum().sort_values()

nyc_complete.columns = nyc_complete.columns.str.replace(r'[ \(\)\-]', '_', regex = True)
nyc_complete.columns

nyc_temperature_complete.columns = nyc_temperature_complete.columns.str.replace(r'[ \(\)\-\/]', '_', regex = True)
nyc_temperature_complete.columns

nyc_complete["UTC_Timestamp__Interval_Ending_"] = pd.to_datetime(nyc_complete["UTC_Timestamp__Interval_Ending_"], format = "%Y-%m-%d %H:%M:%S")
nyc_temperature_complete["UTC_Timestamp__Interval_Ending_"] = pd.to_datetime(nyc_temperature_complete["UTC_Timestamp__Interval_Ending_"], format = "%Y-%m-%d %H:%M:%S")

nyc_ny = nyc_complete[["UTC_Timestamp__Interval_Ending_", "J___New_York_City_Actual_Load__MW_"]]
nyc_ny.columns = ["UTC_Timestamp__Interval_Ending_", "New_York_City_Actual_Load__MW_"]
nyc_ny.describe()

nyc_ny_temperature = nyc_temperature_complete[["UTC_Timestamp__Interval_Ending_", "New_York_City___JFK_Airport_Temperature__Fahrenheit_"]]
nyc_ny_temperature.columns = ["UTC_Timestamp__Interval_Ending_", "New_York_City___JFK_Airport_Temperature__Fahrenheit_"]
nyc_ny_temperature.describe()

# ### Include Full Date Range

# +
datetime_index = pd.DataFrame(pd.date_range(start = nyc_ny["UTC_Timestamp__Interval_Ending_"].min(),
                               end = nyc_ny["UTC_Timestamp__Interval_Ending_"].max(),
                               freq = "1h"))
datetime_index.columns = ["UTC_Timestamp"]

temp_nyc = datetime_index.merge(nyc_ny, how = "left", left_on = "UTC_Timestamp", right_on = "UTC_Timestamp__Interval_Ending_")

temp_nyc.index = temp_nyc["UTC_Timestamp"]
temp_nyc.drop(["UTC_Timestamp", "UTC_Timestamp__Interval_Ending_"], axis = 1, inplace= True)
nyc_ny = temp_nyc

# +
datetime_index = pd.DataFrame(pd.date_range(start = nyc_ny_temperature["UTC_Timestamp__Interval_Ending_"].min(),
                               end = nyc_ny_temperature["UTC_Timestamp__Interval_Ending_"].max(),
                               freq = "1h"))
datetime_index.columns = ["UTC_Timestamp"]

temp_nyc_temp = datetime_index.merge(nyc_ny_temperature, how = "left", left_on = "UTC_Timestamp", right_on = "UTC_Timestamp__Interval_Ending_")

temp_nyc_temp.index = temp_nyc_temp["UTC_Timestamp"]
temp_nyc_temp.drop(["UTC_Timestamp", "UTC_Timestamp__Interval_Ending_"], axis = 1, inplace= True)
nyc_ny_temperature = temp_nyc_temp
# -

nyc_ny = pd.merge(nyc_ny, nyc_ny_temperature, on="UTC_Timestamp", how="left")
print(nyc_ny.shape)
nyc_ny.info()

# ### Define Train-Test Split

# +
nyc_ny.reset_index(inplace=True)
nyc_ny = nyc_ny.sort_values(by = "UTC_Timestamp")

nyc_ny_train = nyc_ny[nyc_ny["UTC_Timestamp"].dt.year < 2024]
nyc_ny_test = nyc_ny[nyc_ny["UTC_Timestamp"].dt.year >= 2024]

print(nyc_ny_train.shape, nyc_ny_test.shape)
# -

nyc_ny_train.tail(1)

nyc_ny_test.head(1)

nyc_ny_train.describe()

nyc_ny_test.describe()


# ### Impute Missing Values for Train Set

def impute_missing_by_seasonal_average(df):
    for i in range(len(df)):
        if pd.isna(df.loc[df.index[i], "New_York_City_Actual_Load__MW_"]):
            # Find all previous similar periods (same month, day, and hour) for load
            similar_periods = df[(df['month'] == df['month'].iloc[i]) &
                                 (df['day'] == df['day'].iloc[i]) &
                                 (df['hour'] == df['hour'].iloc[i]) &
                                 (df.index < df.index[i])]
            if not similar_periods.empty:
                df.loc[df.index[i], "New_York_City_Actual_Load__MW_"] = similar_periods["New_York_City_Actual_Load__MW_"].mean()
        
        if pd.isna(df.loc[df.index[i], "New_York_City___JFK_Airport_Temperature__Fahrenheit_"]):
            # Find all previous similar periods (same month, day, and hour) for temperature
            similar_periods = df[(df['month'] == df['month'].iloc[i]) &
                                 (df['day'] == df['day'].iloc[i]) &
                                 (df['hour'] == df['hour'].iloc[i]) &
                                 (df.index < df.index[i])]
            if not similar_periods.empty:
                df.loc[df.index[i], "New_York_City___JFK_Airport_Temperature__Fahrenheit_"] = similar_periods["New_York_City___JFK_Airport_Temperature__Fahrenheit_"].mean()
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
nyc_ny_train_daily = nyc_ny_train.resample("D").sum()

nyc_ny_train.to_csv("../data/nyc_ny_train_hourly_interpolated.csv")
# -

print(nyc_ny_train_daily.shape)
nyc_ny_train_daily.head()

nyc_ny_test.index = nyc_ny_test["UTC_Timestamp"]
nyc_ny_test = nyc_ny_test.drop(["UTC_Timestamp"], axis = 1)
nyc_ny_test_daily = nyc_ny_test.resample("D").sum()
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
plt.figure(figsize = (20, 10))
sns.lineplot(data = nyc_ny_train_daily, x = nyc_ny_train_daily.index, y = "New_York_City_Actual_Load__MW_").set(title = "NYC Daily Load for Training Set")
plt.xticks(rotation = 45)

plt.savefig("../artifacts/nyc-training-load.png")
plt.show()
# -

plt.figure(figsize = (20, 10))
sns.lineplot(data = nyc_ny_test_daily, x = nyc_ny_test_daily.index, y = "New_York_City_Actual_Load__MW_").set(title = "NYC Daily Load for Test Set")
plt.xticks(rotation = 45)
plt.show()
