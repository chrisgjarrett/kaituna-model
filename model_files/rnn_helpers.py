import numpy as np

def window_reshape_for_rnn(X_orig, n_timesteps):
    """Reshapes and windows an array into sequences of length n_timesteps, for an RNN"""
    X = np.empty((X_orig.shape[0], n_timesteps, X_orig.shape[1]))
    X[:]=np.nan
    for j in range(X_orig.shape[1]):
        for i in range(n_timesteps, X_orig.shape[0]):
            X[i,:n_timesteps, j] = X_orig.iloc[i-n_timesteps:i,j]

    return X[n_timesteps:,:,:]
