import numpy as np
from datetime import datetime, timedelta
import json
import pytz                                      
import pickle as pk

import boto3
from keras.models import load_model

from web_scraper import kaituna_web_scraper, rainfall_forecast_scraper
from preprocessing import aggregate_hourly_data
from model_files.model_definition import mapping_to_target_range

# Days to show historical data for, excluding today
DAYS_TO_GO_BACK = 3
DAYS_TO_PREDICT = 3
GATE_RESOLUTION_LEVEL = 100 # todo this shouldn't be controllable here
GRAPH_DATE_DISPLAY_FORMAT = '%a %d' # 3 letter day and date
DATE_INDEX_FORMAT = "%Y-%m-%d" # String format for indexing the dataframes
TARGET_VARIABLE = "AverageGate"

def handle_predictions(event, context):
    """This gets deployed to AWS to make predictions and update the json data in the bucket"""

    # columns needed: rainfall 3 day's forecast, today's flow, today's rainfall, today's lakelevel

    # Scrape data
    date_today_obj = datetime.now(pytz.timezone('nz'))
    start_date_obj = date_today_obj - timedelta(days=DAYS_TO_GO_BACK) #todo: can this be done better? - perhaps a separate web scraper?
    date_today = date_today_obj.strftime(DATE_INDEX_FORMAT)
    start_date = start_date_obj.strftime(DATE_INDEX_FORMAT)
    
    hourly_data = kaituna_web_scraper.collate_kaituna_data(start_date, date_today)

    # String format
    # Generate average gate ordinal    
    daily_kaituna_data = aggregate_hourly_data.aggregate_hourly_data(hourly_data)

    # Scrape rainfall data
    rainfall_df = rainfall_forecast_scraper.get_rain_forecast(DAYS_TO_PREDICT)
    rainfall_df = rainfall_df.drop(["Date"], axis=1)
    rainfall_df = rainfall_df.transpose()
    rainfall_df.reset_index(drop=True, inplace=True)
    
    rainfall_df.columns = ["Rainfall_lead_1", "Rainfall_lead_2", "Rainfall_lead_3"]

    X = rainfall_df.copy(deep=True)

    # Aggregate data
    X["AverageGate"] = daily_kaituna_data["AverageGate"].loc[date_today]
    X["Rainfall"] = daily_kaituna_data["Rainfall"].loc[date_today]
    X["LakeLevel"] = daily_kaituna_data["LakeLevel"].loc[date_today]

    # Preprocess
    with open('preprocessing/preprocessor.pkl', 'rb') as f:
        preprocessor = pk.load(f)

    X_preprocessed = preprocessor.transform(X)

    # Predict
    model = load_model("model_files/saved_model/", custom_objects={'activation_function': mapping_to_target_range})
    
    rnn_input_timesteps = 1 # todo: shouldn't be here
    #X_preprocessed = np.reshape(np.array(X_preprocessed), (X_preprocessed.shape[0], rnn_input_timesteps, X_preprocessed.shape[1]))
    predicted_gate_levels = model.predict(X_preprocessed)

    # Assemble to json file #todo put this in a method
    historical_sub_dict = {}
    prediction_sub_dict = {}
    predictions_dict = {
        "HistoricalData" : {},
        "PredictedData": {},
        "LastUpdated": {}
    }

    for idx, prediction in enumerate(predicted_gate_levels[0,:]):
        prediction_date_object = date_today_obj + timedelta(days=idx+1)
        prediction_date_display = prediction_date_object.strftime(GRAPH_DATE_DISPLAY_FORMAT)
        prediction_sub_dict[prediction_date_display] = prediction

    predictions_dict["PredictedData"] = prediction_sub_dict

    # Assemble historical flows
    for idx in range(DAYS_TO_GO_BACK,0, -1):
        historical_date_object = date_today_obj - timedelta(days=idx)
        historical_date_display = historical_date_object.strftime(GRAPH_DATE_DISPLAY_FORMAT)
        historical_date_index = historical_date_object.strftime(DATE_INDEX_FORMAT)
        historical_flow = daily_kaituna_data[TARGET_VARIABLE].loc[historical_date_index]
        historical_sub_dict[historical_date_display] = historical_flow

    historical_sub_dict["Today"] = daily_kaituna_data.loc[date_today][TARGET_VARIABLE]
    predictions_dict["HistoricalData"] = historical_sub_dict

    # Add the time it is being updated at
    predictions_dict["LastUpdated"] = date_today_obj.replace(second=0, microsecond=0).strftime('%Y-%m-%d %H:%M %Z')
    
    # Convert Dictionary to JSON String
    data_string = json.dumps(predictions_dict, indent=2, default=str)

    # Upload JSON String to an S3 bucket
    # Try and authenticate with local secrets file
    try:
        #Creating Session With Boto3.
        session = boto3.Session(
        aws_access_key_id=event["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=event["AWS_ACCESS_SECRET_KEY"]
        )
    # Try and authenticate without
    except:
        session = boto3.Session()

    #Creating S3 Resource From the Session.
    s3 = session.resource('s3')
    object = s3.Object(event["BUCKET_NAME"], 'data.json')
    object.put(Body=data_string)

if __name__=="__main__":
    with open("secrets.json") as f:
        secrets = json.load(f)
    handle_predictions(secrets, 2)


