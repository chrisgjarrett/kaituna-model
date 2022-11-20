from sklearn.preprocessing import FunctionTransformer
import numpy as np
import pandas as pd

# Create features and shift output
def make_lags(ts, lags, lead_time=1):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(lead_time, lags + lead_time)
        },
        axis=1)

# Create features and shift output
def make_leads(ts, leads, lag_time=1):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(-i)
            for i in range(lag_time, leads + lag_time)
        },
        axis=1)

def make_multistep_target(ts, steps):
    return pd.concat(
        {f'y_step_{i + 1}': ts.shift(-i)
         for i in range(steps)},
        axis=1)

def make_lags_transformer(n_lags):
    return FunctionTransformer(lambda x: make_lags(x, n_lags))

def make_leads_transformer(n_leads):
    return FunctionTransformer(lambda x: make_leads(x, n_leads))

# Encode these as sine/cosine features to capture cyclical nature
def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

def data_shift_transformer():
    return FunctionTransformer(lambda x: x.dropna())