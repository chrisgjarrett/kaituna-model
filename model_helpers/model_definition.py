import tensorflow as tf
import keras
from keras import layers

def create_model(
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