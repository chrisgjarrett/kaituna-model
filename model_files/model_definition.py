import tensorflow as tf
import keras
from keras import layers

def create_ann(
    input_size,
    output_size,
    learning_rate,
    ):

    model = keras.Sequential([
    layers.Dense(units=100, activation='relu', input_shape=([input_size])), #todo make this not a magic number
    layers.Dropout(0.1),
    layers.Dense(units=50, activation='relu'),
    layers.Dense(units=25, activation='relu'),
    layers.Dense(units=output_size),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(epsilon=learning_rate),
        loss='mean_squared_error',
    )

    return model

def create_rnn(batch_size, time_steps, n_features, output_size, learning_rate=0.01):

    model = keras.Sequential([
        layers.LSTM(units=100, activation='relu', input_shape=(time_steps, n_features), stateful=False,
        return_sequences=True),
        layers.LSTM(units=50, activation='relu', stateful=False),
        layers.Dense(units=output_size),
        ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(epsilon=learning_rate),
        loss='mean_squared_error',
    )

    return model

    
class LSTM_reset_callback(tf.keras.callbacks.Callback):
   def on_epoch_end(self, epoch, logs=None):
        lstm_layer.reset_states()