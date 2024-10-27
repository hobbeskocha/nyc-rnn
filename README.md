# Forecasting NYC Electricity Load using Deep Learning

## Instructions

For ease of version control, the Python notebooks have been saved as Python scripts. HTML views of the notebooks are available [here](html/html_preview.md). Alternatively, the CLI commands [here](cli-reference.md) can convert the Python scripts back to noteboks.

## Overview

This is an personal project conducted by [Ayush Shrestha](https://www.linkedin.com/in/ayush-yoshi-shrestha/). This project analyzes hourly electricity demand and temperature data for the city of New York from June 2021 to September 2024 using recurrent neural networks (RNN) built using PyTorch as well as XGBoost models. The dataset, obtained from the U.S. Energy Information Administration (EIA), can be accessed [here](https://www.eia.gov/electricity/wholesalemarkets/data.php?rto=nyiso)

## Business Objective

For a city as complex as New York, accurately predicting hourly electricity demand is crucial for ensuring smoother, more sustainable, and cost-efficient energy management. The task, however, presents significant challenges due to factors such as seasonality, time dependencies, and external variables like temperature and holidays, which can have a substantial impact on demand.

Effective demand forecasting enables grid operators to prevent overloads, reduce the risk of blackouts, and maintain a balanced power supply, particularly during peak hours or extreme weather events. This allows utility companies to optimize electricity generation, distribution, and storage, ensuring that sufficient power is available to meet demand. Furthermore, precise demand predictions support city planners in shaping future infrastructure, guiding energy policies, and driving sustainability initiatives.

## Dataset Overview

The original dataset comprises over 27,000 hourly observations, covering the period from June 1, 2021, to September 20, 2024. Each year’s data is provided in a separate CSV file, containing hourly timestamps (in the UTC time zone), electricity demand measurements in megawatts (MW) for seven regions within New York State, and temperature measurements in Fahrenheit recorded at Buffalo Niagara and JFK Airport. For the purposes of this analysis, only the demand data specific to New York City area was used.

## Data Processing

**Missing Timestamps**:
I started by importing and merging all the CSV files to create a single comprehensive dataset. Although the dataset initially appeared to be complete with no missing values, further inspection revealed several missing timestamps, ranging from individual hours to multiple days. To address this, a separate list of the full range of dates was generated and used to re-index with the original dataset. This ensured that the entire date range was included, which naturally introduced missing values for the timestamps that were absent in the original data. 

At this stage, the data was split into the training and test sets, using data from 2021 to 2023 for training and data from 2024 as test. This split was conducted early in the process to prevent any cross-contamination between the training and test data during subsequent data cleaning and transformations.


**Load Imputation**:
In the training set, missing values were initially filled using the average demand from the same period in previous years. This approach was chosen due to the data spanning only three years, with no obvious trend. For cases where no prior periods were available for reference, such as missing data in 2021, linear time-based interpolation was applied. In contrast for the test set, to prevent data leakage, a simpler forward-fill method was used to impute missing values. 

## Exploratory Data Analysis (EDA)

To better understand the data, I created boxplots across various time scales (hourly, monthly, quarterly, and yearly) to examine the distribution and variability of electricity demand and temperature. From this we see that peak load occurs around 2100-2200 UTC i.e., 5-6 p.m. EST, and that there are some outliers in the data.

![NYC Hourly](artifacts/nyc-box-hour.png)
![NYC Monthly](artifacts/nyc-box-month.png)

Additionally, I generated histograms to visualize the statistical spread and distribution of the load values. From this plot we also see that the load demand is right-skewed, verifying the presence of outliers.

![NYC Hour](artifacts/nyc-hist-load-hourly.png)

I then plotted a line chart of electricity demand over time, allowing me to identify general temporal patterns and trends in the data. This provided insights into the seasonality and typical range of the data.

![NYC Line](artifacts/nyc-training-load.png)

## Model Configuration

**Data Normalization**:
The training and test sets were normalized by scaling the feature values between 0 and 1. This normalization was crucial to help the models converge more effectively during training by standardizing the input range. Additionally since XGBoost doesn't support time-series forecasting by default, 24 lag features were created for both the load and temperature to allow predicting the current demand. 

**Model Selection**:
To effectively capture long-term dependencies and sequential patterns in the time-series data, both Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks were selected for forecasting future electricity demand. 

Following object-oriented design principles, I defined classes to manage both the dataset and the model architecture. The LSTM and GRU architecture consists of the following components:

- Two LSTM/GRU layers with 50 neurons each, designed to capture the temporal dynamics of the data.
- Dropout layers, with a 20% probability, after each LSTM/GRU layer to prevent overfitting. While additional layers and neurons can capture more complex relationships, they also increase the risk of overfitting, which the dropout layers mitigate by performing regularization.
- A Linear layer that condenses the output from the LSTM/GRU layers down to a single value, enabling the model to produce a continuous regression output for electricity demand predictions.

To provide a performance benchmark against the deep learning models, an XGBoost model was implemented as a robust, yet traditional, alternative. The model was configured with:

- 100 weak learners, allowing it to incrementally build decision trees for enhanced predictive power.
- Squared error as the objective function, allowing each weak learner to minimize prediction error with each iteration.

## Model Training and Evaluation

Using PyTorch's DataLoader, I implemented a training loop with the following key hyperparameters:

- A sliding window size of 24 hours, where each set of 24 hours is used to predict the 25th hour.
- A batch size of 32.
- 30 training epochs.
- The Adam optimizer with a learning rate of 0.0001 to adjust the model's weights during backpropagation.
- Mean Squared Error (MSE) as the loss function, suitable for this regression task.

After 30 epochs, the training converged with an MSE of approximately 0.0001 for LSTM and 0.0002 for GRU on the normalized data. For the test set, the inference loop achieved an average MSE of 0.0003 for LSTM and 0.0005 for GRU, again based on the normalized data. 

I then recorded the predicted and actual values into a dataframe and un-normalized the data back to its original units. This allowed for the calculation of the models' Root Mean Squared Error (RMSE), which was 124 MW for LSTM and 126 MW for GRU. In contrast, XGBoost achieved an RMSE of 86 MW. These are compared to an interquartile range of demand between 4900 and 6600 MW, highlighting the accuracy of the models relative to typical demand levels.


## Findings & Conclusion

The plot below illustrates the predicted electricity demand against the actual load for LSTM, GRU, and XGBoost:

![LSTM predicted vs actual](artifacts/nyc-predicted-actual-line.png)

![GRU predicted vs actual](artifacts/gru-predicted-actual-line.png)

![XGBoost predicted vs actual](artifacts/xgb-predicted-actual-line.png)

XGBoost demonstrated better performance compared to the deep learning models, though the latter still achieved high accuracy. This may be due to a few factors, such as:

- Data size limitations: The dataset may be too small for the deep learning models to fully capture the underlying patterns.
- Effective feature engineering: The explicit feature engineering performed for the XGBoost model likely aided its convergence, even though these steps were essential to properly handle time-series regression.

While this outcome differs from initial expectations, it underscores a key insight: deep learning is not a one-size-fits-all solution. A variety of factors, such as data size, feature complexity, and implementation costs, should be carefully considered to determine if deep learning is the best approach for a given use case and if the potential benefits justify the investment over time.

As a final consideration, the models predict the highest demand of about 7300 MW in July, with a peak hourly demand of around 6400 MW occurring at 6 p.m. EST. These predictions indicate that NYC utilities should prioritize reinforcing the grid to handle increased demand, particularly during summer evenings when electricity usage tends to spike. Additionally, allocating extra resources during peak periods could help prevent blackouts or system overloads. Utilities may also consider implementing demand-side management strategies, such as incentivizing consumers to reduce electricity usage during peak hours, especially in the evenings, to maintain grid stability.
