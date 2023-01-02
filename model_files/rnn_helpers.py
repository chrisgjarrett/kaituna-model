import numpy as np


def window_reshape_for_rnn(X_orig, n_timesteps, forecast_horizon):
    """Reshapes and windows an array into sequences of length n_timesteps, for an RNN"""
    X = np.empty((X_orig.shape[0], n_timesteps + forecast_horizon, X_orig.shape[1]))
    X[:]=np.nan
    for feature_id in range(X_orig.shape[1]):
        for sample_id in range(n_timesteps, X_orig.shape[0] - forecast_horizon):
            X[sample_id,:n_timesteps, feature_id] = X_orig.iloc[sample_id-n_timesteps:sample_id,feature_id]

            if (X_orig.columns[feature_id] == "Rainfall"):
                X[sample_id,n_timesteps:, feature_id] = X_orig.iloc[sample_id:sample_id+forecast_horizon,feature_id]
            else:
                X[sample_id,n_timesteps:, feature_id] = 0

    return X[n_timesteps:-forecast_horizon,:,:]
