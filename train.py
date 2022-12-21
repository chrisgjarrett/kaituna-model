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
from sklearn.compose import ColumnTransformer
from matplotlib import pyplot as plt
import seaborn as sns
from tensorflow import saved_model

from helpers.transfomers import make_leads_transformer
from helpers.transfomers import make_multistep_target
from preprocessing import aggregate_hourly_data
from preprocessing import feature_generator
from model_files.model_definition import create_rnn, create_ann, mapping_to_target_range
from model_files.rnn_helpers import window_reshape_for_rnn
from helpers.plotting_helpers import visualise_results

# Configure logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

DAYS_TO_PREDICT = 3
TARGET_VARIABLE = "AverageGate"
GATE_RESOLUTION_LEVEL = 100
TRAINING_DATA_PATH = "datasets/training_data_artifact.csv"
MAX_OUTPUT = 1500
MIN_OUTPUT = 0

# Are we training the final model or running cross-validation?
TRAIN_FINAL_MODEL = True

# Experiment name
experiment_name = 'CNN-RNN hybrid'

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

    # Days to look back
    n_target_lags = 1
    n_rainfall_lags = 1 
    n_lake_level_lags = 1
    n_rainfall_leads = 3 

    # Bundle preprocessing for data.
    lag_generator = ColumnTransformer(
        transformers=[
            #("target_lags", make_lags_transformer(n_target_lags), [target_variable]),
            #("rainfall_lags", make_lags_transformer(n_rainfall_lags), ["Rainfall"]),
            #("lakelevel_lags", make_lags_transformer(n_lake_level_lags), ["LakeLevel"]),
            ("rainfall_leads", make_leads_transformer(n_rainfall_leads), ["Rainfall"]),
        ],
        remainder='passthrough'
    )

    lead_lag_columns = [
        #"Target_lag",
        #"Rainfall_lag",
        #"LakeLevel_lag",
        "Rainfall_lead_1",
        "Rainfall_lead_2",
        "Rainfall_lead_3",
        ]

    # Select data for model fitting
    X_cols_to_lag = daily_kaituna_data[["Rainfall"]]

    # Create lags
    X_lead_lags = pd.DataFrame(lag_generator.fit_transform(X_cols_to_lag),
                        index = daily_kaituna_data.index,
                        columns=lead_lag_columns)

    X_raw = pd.merge(X_lead_lags, daily_kaituna_data, left_index=True, right_index=True)

    # Generate feature set
    X_features = feature_generator.feature_generator(X_raw, TARGET_VARIABLE)
    
    # Create target variable
    if (DAYS_TO_PREDICT == 1):
        y = pd.DataFrame(daily_kaituna_data[TARGET_VARIABLE])
    else:
        y = make_multistep_target(daily_kaituna_data[TARGET_VARIABLE], DAYS_TO_PREDICT) 

    # Drop columns with missing values    
    y = y.dropna()
    y, X = y.align(X_features, join='inner', axis=0)

    # Split into test and train/validation
    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X, y, test_size=0.20, shuffle=False)

    # Save dataframe to file for artifact logging
    training_data_df = pd.concat([X_train_df, y_train_df], axis=1)
    training_data_df.to_csv(TRAINING_DATA_PATH)
    
    # Construct model
    n_epochs = 2000
    learning_rate = 0.1
    patience = n_epochs // 5
    min_delta = 1
    n_timesteps = 7
    batch_size = X_train_df.shape[0] - n_timesteps # Play with this
    n_hidden_layers = 1
    lstm_units = {"layer1":50}
       
    early_stopping = EarlyStopping(
        min_delta=min_delta, # minimium amount of change to count as an improvement
        patience=patience, # how many epochs to wait before stopping
        restore_best_weights=True,
        monitor='loss'
    )

#input_shape=(n_steps, 1, n_length, n_features)
    wrapped_model = KerasRegressor(
            create_rnn,
            input_shape=(n_timesteps, X_train_df.shape[1]),
            output_size=y_train_df.shape[1],
            learning_rate=learning_rate,
            max_output = MAX_OUTPUT,
            min_output = MIN_OUTPUT,
            epochs=n_epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=True, #state_reset_callback
        )

    # Reshape for RNN
    # Window the data into lstm form
    ctr = 0
    X_train_df = window_reshape_for_rnn(X_train_df, n_timesteps)
    X_test_df = window_reshape_for_rnn(X_test_df, n_timesteps)

    # Align the X and y
    y_train_df = y_train_df.iloc[n_timesteps:,:]
    y_test_df = y_test_df.iloc[n_timesteps:,:]

    # If we want to just train the model, rather than perform cross-validation
    if (TRAIN_FINAL_MODEL == True):

        final_model = wrapped_model.fit(
            X_train_df,
            y_train_df,
            validation_data=(X_test_df, y_test_df),
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

        y_fit = pd.DataFrame(final_model.model.predict(X_train_df), index=y_train_df.index, columns=y_train_df.columns)
        y_pred = pd.DataFrame(final_model.model.predict(X_test_df), index=y_test_df.index, columns=y_test_df.columns)

        visualise_results(daily_kaituna_data[TARGET_VARIABLE], y_fit, y_pred)

        # Save model files
        final_model.model.save("model_files/saved_model/")
        
        exit()

    # Cross-validation
    #todo run this multiple times and do statistics
    # to try: lstm complexities, hybrid cnn, hybrid with dense
    mlflow.set_experiment(experiment_name)

    # Start experiment
    with mlflow.start_run():    

        # Cross validation
        n_splits=5
        tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=n_splits)
        scores = cross_val_score(
            wrapped_model,
            X_train_df_rnn,
            y_train_df,
            cv=tscv,
            scoring='neg_mean_squared_error',
            fit_params={'validation_data':(X_test_df_rnn, y_test_df)}
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
        mlflow.log_param('time_steps',n_timesteps)
        mlflow.log_param('stateful', False)
        mlflow.log_param("n_hidden_layers", n_hidden_layers)
        mlflow.log_param("units", lstm_units)
        mlflow.log_param("batch_size", batch_size)

        # Save model and data set
        mlflow.sklearn.log_model(
        sk_model=wrapped_model,
        artifact_path="model",
        )

        mlflow.log_artifacts('datasets', TRAINING_DATA_PATH)