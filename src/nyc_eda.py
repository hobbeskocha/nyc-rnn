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

import numpy as np
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

nyc_complete.columns = nyc_complete.columns.str.replace(r'[ \(\)\-]', '_', regex=True)
nyc_complete.columns

nyc_complete["UTC_Timestamp__Interval_Ending_"] = pd.to_datetime(nyc_complete["UTC_Timestamp__Interval_Ending_"], format = "%Y-%m-%d %H:%M:%S")

nyc_ny = nyc_complete[["UTC_Timestamp__Interval_Ending_", "J___New_York_City_Actual_Load__MW_"]]
nyc_ny.head()

nyc_ny.columns = ["UTC_Timestamp__Interval_Ending_", "New_York_City_Actual_Load__MW_"]
nyc_ny.describe()

# +
nyc_ny_train = nyc_ny[nyc_ny["UTC_Timestamp__Interval_Ending_"].dt.year < 2024]
nyc_ny_test = nyc_ny[nyc_ny["UTC_Timestamp__Interval_Ending_"].dt.year >= 2024]

print(nyc_ny_train.shape, nyc_ny_test.shape)
print(nyc_ny_train.tail(1))
print(nyc_ny_test.head(1))

# +
datetime_index = pd.DataFrame(pd.date_range(start = nyc_ny_train["UTC_Timestamp__Interval_Ending_"].min(),
                               end = nyc_ny_train["UTC_Timestamp__Interval_Ending_"].max(),
                               freq = "1h"))
datetime_index.columns = ["UTC_Timestamp"]

temp = datetime_index.merge(nyc_ny_train, how = "left", left_on = "UTC_Timestamp", right_on = "UTC_Timestamp__Interval_Ending_")

temp.index = temp["UTC_Timestamp"]
temp.drop(["UTC_Timestamp", "UTC_Timestamp__Interval_Ending_"], axis = 1, inplace= True)

temp_interpolate = temp.interpolate(method = "time")

print(temp_interpolate.info())
temp_interpolate.head()
# -

temp.to_csv("../data/nyc_ny_train_hourly.csv")
temp_interpolate.to_csv("../data/nyc_ny_train_hourly_interpolated.csv")

# +
nyc_ny_train = temp_interpolate
nyc_ny_train_daily = nyc_ny_train.resample("D").sum()

print(nyc_ny_train_daily.shape)
nyc_ny_train_daily.head()
# -

# ### Data Visualization

# +
sns.boxplot(nyc_ny_train_daily).set(title = "NYC Daily")
plt.show()

sns.boxplot(nyc_ny_train).set(title = "NYC Hourly")
plt.show()

# +
sns.histplot(nyc_ny_train_daily).set(title = "NYC Daily")
plt.show()

sns.histplot(nyc_ny_train).set(title = "NYC Hourly")
plt.show()
# -

sns.lineplot(data = nyc_ny_train_daily, x = nyc_ny_train_daily.index, y = "New_York_City_Actual_Load__MW_")
plt.xticks(rotation = 45)
plt.show()
