# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: winery-project
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

# ## Merge Dataframes

print(list(nyc_2021.columns) == list(nyc_2022.columns) 
      and list(nyc_2022.columns) == list(nyc_2023.columns) 
      and list(nyc_2023.columns) == list(nyc_2024.columns))

# +
nyc_complete = pd.concat([nyc_2021, nyc_2022, nyc_2023, nyc_2024], axis = 0)
print(nyc_complete.shape)

nyc_complete.head()
# -

nyc_complete.to_csv("../data/nyc_complete.csv", index = False)

# ## Exploratory Data Analysis

nyc_complete.info()

nyc_complete.isna().sum().sort_values()

nyc_complete.columns = nyc_complete.columns.str.replace(r'[ \(\)\-]', '_', regex = True)
nyc_complete.columns

nyc_complete["UTC_Timestamp__Interval_Ending_"] = pd.to_datetime(nyc_complete["UTC_Timestamp__Interval_Ending_"], format = "%Y-%m-%d %H:%M:%S")

nyc_ny = nyc_complete[["UTC_Timestamp__Interval_Ending_", "J___New_York_City_Actual_Load__MW_"]]
nyc_ny.head()

nyc_ny.columns = ["UTC_Timestamp__Interval_Ending_", "New_York_City_Actual_Load__MW_"]
nyc_ny.describe()

# ### Define Train-Test Split

# +
nyc_ny = nyc_ny.sort_values(by = "UTC_Timestamp__Interval_Ending_")

nyc_ny_train = nyc_ny[nyc_ny["UTC_Timestamp__Interval_Ending_"].dt.year < 2024]
nyc_ny_test = nyc_ny[nyc_ny["UTC_Timestamp__Interval_Ending_"].dt.year >= 2024]

print(nyc_ny_train.shape, nyc_ny_test.shape)
print(nyc_ny_train.tail(1))
print(nyc_ny_test.head(1))
# -

nyc_ny_train.describe()

nyc_ny_test.describe()


# ### Include Full Date Range and Impute Missing Values for Train Set

def impute_missing_by_seasonal_average(df):
    for i in range(len(df)):
        if pd.isna(df.loc[df.index[i], "New_York_City_Actual_Load__MW_"]):
            # Find all previous similar periods (same month, day, and hour)
            similar_periods = df[(df['month'] == df['month'].iloc[i]) &
                                 (df['day'] == df['day'].iloc[i]) &
                                 (df['hour'] == df['hour'].iloc[i]) &
                                 (df.index < df.index[i])]
            
            if not similar_periods.empty:
                df.loc[df.index[i], "New_York_City_Actual_Load__MW_"] = similar_periods["New_York_City_Actual_Load__MW_"].mean()
    return df


# +
datetime_index = pd.DataFrame(pd.date_range(start = nyc_ny_train["UTC_Timestamp__Interval_Ending_"].min(),
                               end = nyc_ny_train["UTC_Timestamp__Interval_Ending_"].max(),
                               freq = "1h"))
datetime_index.columns = ["UTC_Timestamp"]

temp = datetime_index.merge(nyc_ny_train, how = "left", left_on = "UTC_Timestamp", right_on = "UTC_Timestamp__Interval_Ending_")

temp.index = temp["UTC_Timestamp"]
temp.drop(["UTC_Timestamp", "UTC_Timestamp__Interval_Ending_"], axis = 1, inplace= True)

temp["month"] = temp.index.month
temp["day"] = temp.index.day
temp["hour"] = temp.index.hour

temp_interpolate = impute_missing_by_seasonal_average(temp)
temp_interpolate.interpolate(method = "time", inplace = True)

temp_interpolate.drop(columns=["month", "day", "hour"], inplace=True)

print(temp_interpolate.info())
temp_interpolate.head()

# +
nyc_ny_train = temp_interpolate
nyc_ny_train_daily = nyc_ny_train.resample("D").sum()

nyc_ny_train.to_csv("../data/nyc_ny_train_hourly_interpolated.csv")
# -

print(nyc_ny_train_daily.shape)
nyc_ny_train_daily.head()

# ### Include Full Date Range for Test Set

# +
datetime_index = pd.DataFrame(pd.date_range(start = nyc_ny_test["UTC_Timestamp__Interval_Ending_"].min(),
                               end = nyc_ny_test["UTC_Timestamp__Interval_Ending_"].max(),
                               freq = "1h"))
datetime_index.columns = ["UTC_Timestamp"]
temp = datetime_index.merge(nyc_ny_test, how = "left", left_on = "UTC_Timestamp", right_on = "UTC_Timestamp__Interval_Ending_")

temp.index = temp["UTC_Timestamp"]
temp.drop(["UTC_Timestamp", "UTC_Timestamp__Interval_Ending_"], axis = 1, inplace= True)

temp.info()
temp.head()
# -

nyc_ny_test = temp
nyc_ny_test_daily = nyc_ny_test.resample("D").sum()
nyc_ny_test.to_csv("../data/nyc_ny_test_hourly.csv")

# ### Data Visualization

nyc_viz = nyc_ny_train.copy()
nyc_viz["year"] = nyc_viz.index.year
nyc_viz["quarter"] = nyc_viz.index.quarter
nyc_viz["month"] = nyc_viz.index.month
nyc_viz["hour"] = nyc_viz.index.hour

# +
sns.boxplot(nyc_viz, x = "hour", y = "New_York_City_Actual_Load__MW_").set(title = "NYC Load by Hour")
plt.show()

sns.boxplot(nyc_viz, x = "month", y = "New_York_City_Actual_Load__MW_").set(title = "NYC Load by Month")
plt.show()

sns.boxplot(nyc_viz, x = "quarter", y = "New_York_City_Actual_Load__MW_").set(title = "NYC Load by Quarter")
plt.show()

sns.boxplot(nyc_viz, x = "year", y = "New_York_City_Actual_Load__MW_").set(title = "NYC Load by Year")
plt.show()

# +
sns.histplot(nyc_ny_train_daily).set(title = "NYC Daily")
plt.show()

sns.histplot(nyc_ny_train).set(title = "NYC Hourly")
plt.show()
# -

plt.figure(figsize = (20, 10))
sns.lineplot(data = nyc_ny_train_daily, x = nyc_ny_train_daily.index, y = "New_York_City_Actual_Load__MW_").set(title = "NYC Daily Load for Training Set")
plt.xticks(rotation = 45)
plt.show()

plt.figure(figsize = (20, 10))
sns.lineplot(data = nyc_ny_test_daily, x = nyc_ny_test_daily.index, y = "New_York_City_Actual_Load__MW_").set(title = "NYC Daily Load for Test Set")
plt.xticks(rotation = 45)
plt.show()
