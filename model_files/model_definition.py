import tensorflow as tf
import keras
from keras import layers, metrics

def create_ann(n_features, output_size, learning_rate=0.01):

    model = keras.Sequential([
        layers.Dense(units=50, activation='relu', input_shape=(n_features,)),
        layers.Dropout(0.15),
        layers.Dense(units=10, activation='relu'),
        layers.Dense(units=output_size),
        ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(epsilon=learning_rate),
        metrics=[metrics.RootMeanSquaredError()],
        loss='mean_squared_error',
    )

    return model

def create_rnn(batch_size, time_steps, n_features, output_size, learning_rate=0.01):

    model = keras.Sequential([
        layers.LSTM(units=50, activation='relu', batch_input_shape=(batch_size, time_steps, n_features), stateful=False, return_sequences=True),
        layers.LSTM(units=25, activation='relu', stateful=False),
        layers.Dense(units=output_size),
        ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(epsilon=learning_rate),
        metrics=[metrics.RootMeanSquaredError()],
        loss='mean_squared_error',
    )

    return model
