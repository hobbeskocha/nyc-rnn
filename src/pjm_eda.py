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

# # PJM: EDA

# ## Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.max_columns = 40
pd.options.display.max_rows = 100

# ## Import Data

pjm_2021 = pd.read_csv("../data/pjm_load_act_hr_2021.csv", header = 3)
pjm_2022 = pd.read_csv("../data/pjm_load_act_hr_2022.csv", header = 3)
pjm_2023 = pd.read_csv("../data/pjm_load_act_hr_2023.csv", header = 3)
pjm_2024 = pd.read_csv("../data/pjm_load_act_hr_2024.csv", header = 3)

# ## Merge Dataframes

print(list(pjm_2021.columns) == list(pjm_2022.columns) 
      and list(pjm_2022.columns) == list(pjm_2023.columns) 
      and list(pjm_2023.columns) == list(pjm_2024.columns))

# +
pjm_complete = pd.concat([pjm_2021, pjm_2022, pjm_2023, pjm_2024], axis = 0)
print(pjm_complete.shape)

pjm_complete.head()
# -

pjm_complete.to_csv("../data/pjm_complete.csv", index = False)

# ## Exploratory Data Analysis

pjm_complete.info()

pjm_complete.isna().sum().sort_values()

pjm_complete.columns = pjm_complete.columns.str.replace(r'[ \.\(\)\-]', '_', regex=True)
pjm_complete.columns

pjm_complete["UTC_Timestamp__Interval_Ending_"] = pd.to_datetime(pjm_complete["UTC_Timestamp__Interval_Ending_"], format = "%Y-%m-%d %H:%M:%S")

pjm_baltimore = pjm_complete[["UTC_Timestamp__Interval_Ending_", "Baltimore_Gas_and_Electric_Company_Actual_Load__MW_"]]
pjm_baltimore.head()

pjm_baltimore.describe()

# +
datetime_index = pd.DataFrame(pd.date_range(start=pjm_baltimore["UTC_Timestamp__Interval_Ending_"].min(),
                               end=pjm_baltimore["UTC_Timestamp__Interval_Ending_"].max(),
                               freq="1h"))
datetime_index.columns = ["UTC_Timestamp"]

temp = datetime_index.merge(pjm_baltimore, how = "left", left_on = "UTC_Timestamp", right_on = "UTC_Timestamp__Interval_Ending_")
print(temp.info())
temp.head()

# pjm_baltimore_daily = pjm_baltimore.drop(["UTC_Timestamp__Interval_Ending_"], axis = 1)
# pjm_baltimore_daily = pjm_baltimore_daily.resample("D").sum()

# print(pjm_baltimore_daily.shape)
# pjm_baltimore_daily.head()
# -

# ### Data Visualization

# +
sns.boxplot(pjm_baltimore_daily).set(title = "Baltimore Daily")
plt.show()

sns.boxplot(pjm_baltimore).set(title = "Baltimore Hourly")
plt.show()

# +
sns.histplot(pjm_baltimore_daily).set(title = "Baltimore Daily")
plt.show()

sns.histplot(pjm_baltimore).set(title = "Baltimore Daily")
plt.show()
# -

sns.lineplot(data = pjm_baltimore_daily, x = "UTC_Timestamp__Interval_Ending_", y = "Baltimore_Gas_and_Electric_Company_Actual_Load__MW_")
plt.show()
