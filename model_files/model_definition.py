import tensorflow as tf
import keras
from keras import layers, metrics
from keras import backend as BK


def mapping_to_target_range( x, target_min, target_max) :
    x02 = BK.tanh(x) + 1 # x in range(0,2)
    scale = ( target_max-target_min )/2.
    return  x02 * scale + target_min

def create_ann(input_shape, output_size, learning_rate=0.01,  max_output=1500, min_output=0):

    model = keras.Sequential([
        layers.Dense(units=50, activation='relu', input_shape=(input_shape)),
        layers.Dropout(0.05),
        layers.Dense(units=25, activation='relu'),
        layers.Dropout(0.05),
        layers.Dense(units=12, activation='relu'),
        layers.Dense(units=output_size),
        layers.Lambda(lambda x: mapping_to_target_range(x, target_max=max_output, target_min=min_output))
        ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(epsilon=learning_rate),
        metrics=[metrics.RootMeanSquaredError()],
        loss='mean_squared_error',
    )

    return model

def create_rnn(input_shape, output_size, learning_rate=0.05, max_output=1500, min_output=0):

    model = keras.Sequential([
        #layers.Conv1D(filters = 32, kernel_size=3, activation='relu', input_shape=input_shape), 
        layers.Bidirectional(layers.LSTM(units=20, activation='relu', stateful=False, return_sequences=True), input_shape=input_shape),
        # layers.Bidirectional(layers.LSTM(units=20, activation='relu', stateful=False, return_sequences=True)),
        # layers.Bidirectional(layers.LSTM(units=10, activation='relu', stateful=False, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(units=10, activation='relu', stateful=False, return_sequences=False)),
        layers.Dropout(0.15),
        layers.Dense(units=output_size),
        layers.Lambda(lambda x: mapping_to_target_range(x, target_max=max_output, target_min=min_output)),
        ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(epsilon=learning_rate),
        metrics=[metrics.RootMeanSquaredError()],
        loss='mean_squared_error',
    )

    return model
