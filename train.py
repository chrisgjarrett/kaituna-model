# Imports and helper methods

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import logging
import pickle as pk
import mlflow
import mlflow.sklearn
from keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor

from helpers.transfomers import make_multistep_target
from helpers import aggregate_hourly_data
from preprocessing import feature_generator
from model_files.model_definition import create_ann, create_rnn, LSTM_reset_callback

# Configure logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

DAYS_TO_PREDICT = 3
TARGET_VARIABLE = "AverageGate"
GATE_RESOLUTION_LEVEL = 100
TRAINING_DATA_PATH = "datasets/training_data_artifact.csv"

TRAIN_FINAL_MODEL = True

# Load data
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Load data
    hourly_kaituna_data = pd.read_csv('datasets/hourly_kaituna_data.csv',parse_dates=["TimeStamp"])
    hourly_kaituna_data["TimeStamp"] = pd.to_datetime(hourly_kaituna_data['TimeStamp'], utc=True)
    hourly_kaituna_data = hourly_kaituna_data.set_index("TimeStamp")

    # Convert to daily
    daily_kaituna_data = aggregate_hourly_data.aggregate_hourly_data(hourly_kaituna_data)

    # Generate feature set
    X_features = feature_generator.feature_generator(daily_kaituna_data, GATE_RESOLUTION_LEVEL, DAYS_TO_PREDICT, TARGET_VARIABLE)
 
    # Create target variable
    if (DAYS_TO_PREDICT == 1):
        y = pd.DataFrame(daily_kaituna_data[TARGET_VARIABLE])
    else:
        y = make_multistep_target(daily_kaituna_data[TARGET_VARIABLE], DAYS_TO_PREDICT) 

    # Drop columns with missing values    
    y = y.dropna()

    # Align the X and y
    y, X = y.align(daily_kaituna_data, join='inner', axis=0)

    # Split into test and train/validation
    X_train_df, X_test_df, y_train_df, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

    # Save dataframe to file for artifact logging
    training_data_df = pd.concat([X_train_df, y_train_df], axis=1)
    training_data_df.to_csv(TRAINING_DATA_PATH)
    
    # Split into training and validation

    # Construct model
    n_epochs = 5#2000
    learning_rate = 0.01
    patience = 100
    min_delta = 100
    batch_size = 1 # Play with this
    
       
    early_stopping = EarlyStopping(
        min_delta=min_delta, # minimium amount of change to count as an improvement
        patience=patience, # how many epochs to wait before stopping
        restore_best_weights=True,
        monitor='loss'
    )

    #Fit model
    wrapped_model = KerasRegressor(
        create_ann,
        input_size=X_train_df.shape[1],
        output_size=y_train_df.shape[1],
        learning_rate=learning_rate,
        epochs=n_epochs,
        callbacks=[early_stopping],
        #verbose=verbose
        )
    
    state_reset_callback = LSTM_reset_callback()
    wrapped_model = KerasRegressor(
        create_rnn,
        batch_size=batch_size,
        time_steps=1,
        n_features=X_train_df.shape[1],
        output_size=y_train_df.shape[1],
        epochs=n_epochs,
        callbacks=[early_stopping], #state_reset_callback
    )

    # Reshape for RNN
    X_train_reshaped = np.reshape(np.array(X_train_df), (X_train_df.shape[0], batch_size, X_train_df.shape[1]))

    if (TRAIN_FINAL_MODEL == True):

        final_model = wrapped_model.fit(X_train_reshaped, y_train_df)
        pk.dump(final_model, open("model_files/model.pkl","wb"))

        exit()

    experiment_name = 'Trying a stateless RNN'
    mlflow.set_experiment(experiment_name)
    
    # Start experiment
    with mlflow.start_run():    
 
        # Cross validation
        n_splits=5
        tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=n_splits)
        scores = cross_val_score(wrapped_model,
                                X_train_reshaped,
                                y_train_df,
                                cv=tscv,
                                scoring='neg_mean_squared_error',
                                #fit_params={'model__callbacks':[early_stopping]}
                                )
        scores = np.sqrt(-scores)

        # Logging metrics
        for idx, score in enumerate(scores):
            mlflow.log_metric("cross_val_rmse".join(["_", str(idx)]), score)
        
        mlflow.log_metric("median_score", np.median(scores))
        mlflow.log_param("n_epochs", n_epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("early_stopping_patience", patience)
        mlflow.log_param("early_stopping_min_delta", min_delta)
        mlflow.log_param('time_steps',1)
        mlflow.log_param('stateful', False)
        # Save model and data set
        mlflow.sklearn.log_model(
        sk_model=wrapped_model,
        artifact_path="model",
        )

        mlflow.log_artifacts('datasets', TRAINING_DATA_PATH)

