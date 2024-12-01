import torch.nn as nn
from statsmodels.tsa.stattools import adfuller

def init_weights(m):
    if isinstance(m, nn.Linear):
        # Apply Xavier initialization to Linear layers
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
        # Apply Xavier initialization to LSTM or GRU weights
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

def is_stationary(time_series, significance = 0.05):
    print(time_series.name)
    adf = adfuller(time_series, autolag = "AIC")
    test_stat = adf[0]
    p_val = adf[1]

    print("ADF Statistic: ", test_stat)
    print("ADF p-value: ", p_val)

    return True if p_val < significance else False