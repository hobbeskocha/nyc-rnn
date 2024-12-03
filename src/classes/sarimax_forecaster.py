from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SarimaxForecaster(BaseEstimator, RegressorMixin):
    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, X, y):
        self.model_ = SARIMAX(y, exog = X, order = self.order, seasonal_order=self.seasonal_order)
        self.results_ = self.model_.fit(disp = False)
        return self
    
    def predict(self, X):
        return self.results_.forecast(steps = len(X), exog = X)