# Imports and helper methods

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import mlflow
import mlflow.sklearn
from keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor
from matplotlib import pyplot as plt

from helpers.transfomers import make_multistep_target
from preprocessing import aggregate_hourly_data
from preprocessing import feature_generator
from model_files.model_definition import create_rnn
from model_files.rnn_helpers import window_reshape_for_rnn
from helpers.plotting_helpers import visualise_results

# Constants
DAYS_TO_PREDICT = 3
TARGET_VARIABLE = "AverageGate"
GATE_RESOLUTION_LEVEL = 100
TRAINING_DATA_PATH = "datasets/training_data_artifact.csv"
MAX_OUTPUT = 1500
MIN_OUTPUT = 0
N_TRIALS = 5 # Number of times to cross-validate

# Are we training the final model or running cross-validation?
TRAIN_FINAL_MODEL = False

# Experiment name
experiment_name = '2 layers, learning rate'

# Load data
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Load data
    hourly_kaituna_data = pd.read_csv('datasets/hourly_kaituna_data.csv',parse_dates=["TimeStamp"])
    hourly_kaituna_data["TimeStamp"] = pd.to_datetime(hourly_kaituna_data['TimeStamp'], utc=True)
    hourly_kaituna_data = hourly_kaituna_data.set_index("TimeStamp")

    # Convert to daily summaries
    daily_kaituna_data = aggregate_hourly_data.aggregate_hourly_data(hourly_kaituna_data)

    # Preprocessing data to create features
    X_features = feature_generator.feature_generator(daily_kaituna_data, TARGET_VARIABLE)
    
    # Create target variable
    y = make_multistep_target(daily_kaituna_data[TARGET_VARIABLE], DAYS_TO_PREDICT) 

    # Drop columns with missing values from shifting    
    y = y.dropna()
    y, X = y.align(X_features, join='inner', axis=0)

    # Split into test and train/validation
    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X, y, test_size=0.33, shuffle=False)

    # Save dataframe to file for artifact logging
    training_data_df = pd.concat([X_train_df, y_train_df], axis=1)
    training_data_df.to_csv(TRAINING_DATA_PATH)
    
    # Construct model
    n_epochs = 2000
    learning_rate = 0.0001
    patience = n_epochs // 5
    min_delta = 1
    n_timesteps = 1
    batch_size = X_train_df.shape[0] - n_timesteps
    
    # Configure early stopping criteria
    early_stopping = EarlyStopping(
        min_delta=min_delta, # minimium amount of change to count as an improvement
        patience=patience, # how many epochs to wait before stopping
        restore_best_weights=True,
        monitor='loss'
    )

    # Create model
    wrapped_model = KerasRegressor(
            create_rnn,
            input_shape=(n_timesteps + DAYS_TO_PREDICT, X_train_df.shape[1],),
            output_size=y_train_df.shape[1],
            learning_rate=learning_rate,
            max_output = MAX_OUTPUT,
            min_output = MIN_OUTPUT,
            epochs=n_epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=True,
        )

    # Reshape for lstm
    ctr = 0
    X_train = window_reshape_for_rnn(X_train_df, n_timesteps, DAYS_TO_PREDICT)
    X_test = window_reshape_for_rnn(X_test_df, n_timesteps, DAYS_TO_PREDICT)

    # Align the X and y.
    y_train_df = y_train_df.iloc[n_timesteps:-DAYS_TO_PREDICT,:]
    y_test_df = y_test_df.iloc[n_timesteps:-DAYS_TO_PREDICT,:]

    # If we want to just train the model, rather than perform cross-validation
    if (TRAIN_FINAL_MODEL == True):
        
        final_model = wrapped_model.fit(
            X_train,
            y_train_df,
            validation_data=(X_test, y_test_df),
            callbacks=[early_stopping],
            epochs=n_epochs)
        
        # Examine history
        history = final_model.history
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show(block=False)

        y_fit = pd.DataFrame(final_model.model.predict(X_train), index=y_train_df.index, columns=y_train_df.columns)
        y_pred = pd.DataFrame(final_model.model.predict(X_test), index=y_test_df.index, columns=y_test_df.columns)

        visualise_results(daily_kaituna_data[TARGET_VARIABLE], y_fit, y_pred)

        # Save model files
        final_model.model.save("model_files/saved_model/")
        
        input("Press enter to exit")

        exit()

    for i in range(N_TRIALS):
        
        # Cross-validation
        mlflow.set_experiment(experiment_name)

        # Start experiment
        with mlflow.start_run():    

            # Cross validation
            n_splits=5
            tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=n_splits)
            scores = cross_val_score(
                wrapped_model,
                X_train,
                y_train_df,
                cv=tscv,
                scoring='neg_mean_squared_error',
                fit_params={'validation_data':(X_test, y_test_df)}
            )

            # Get RMSE
            scores = np.sqrt(-scores)

            # Logging metrics
            for idx, score in enumerate(scores):
                mlflow.log_metric("cross_val_rmse".join(["_", str(idx)]), score)
            
            mlflow.log_metric("median_score", np.median(scores))

            mlflow.log_param("n_epochs", n_epochs)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("early_stopping_patience", patience)
            mlflow.log_param("early_stopping_min_delta", min_delta)
            mlflow.log_param('time_steps',n_timesteps)
            mlflow.log_param("batch_size", batch_size)

            # Save model and data set
            mlflow.sklearn.log_model(
            sk_model=wrapped_model,
            artifact_path="model",
            )

            mlflow.log_artifacts('datasets', TRAINING_DATA_PATH)