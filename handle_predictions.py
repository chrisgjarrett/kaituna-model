import numpy as np
from datetime import datetime, timedelta
import json
import pytz                                      
import pickle as pk

import boto3

from web_scraper import kaituna_web_scraper, rainfall_forecast_scraper
from preprocessing import aggregate_hourly_data

# Days to show historical data for, excluding today
DAYS_TO_GO_BACK = 3
DAYS_TO_PREDICT = 3
GATE_RESOLUTION_LEVEL = 100 # todo this shouldn't be controllable here
target_variable = "AverageGate"

def handle_predictions(event, context):
    """This gets deployed to AWS to make predictions and update the json data in the bucket"""

    # columns needed: rainfall 3 day's forecast, today's flow, today's rainfall, today's lakelevel

    # Scrape data
    date_today_obj = datetime.now(pytz.timezone('nz'))
    start_date_obj = date_today_obj - timedelta(days=DAYS_TO_GO_BACK) #todo: can this be done better? - perhaps a separate web scraper?
    date_today = date_today_obj.strftime('%Y-%m-%d')
    start_date = start_date_obj.strftime('%Y-%m-%d')
    
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
    #X["AverageGate"] = daily_kaituna_data["AverageGate"].loc[date_today]
    X["Rainfall"] = daily_kaituna_data["Rainfall"].loc[date_today]
    X["LakeLevel"] = daily_kaituna_data["LakeLevel"].loc[date_today]

    # Preprocess
    preprocessor = pk.load(open('preprocessing/preprocessor.pkl', 'rb'))
    X_preprocessed = preprocessor.transform(X)

    # Predict #todo: update with actual model
    model = pk.load(open('model_files/model.pkl', 'rb'))
    batch_size = 1 # todo: shouldn't be here
    X_reshaped = np.reshape(np.array(X_preprocessed), (X_preprocessed.shape[0], batch_size, X_preprocessed.shape[1]))
    predicted_gate_levels = model.model.predict(X_reshaped)

    # Assemble to json file #todo put this in a method
    historical_sub_dict = {}
    prediction_sub_dict = {}
    predictions_dict = {
        "HistoricalData" : {},
        "PredictedData": {}
    }

    for idx, prediction in enumerate(predicted_gate_levels[0,:]):
        prediction_date = (date_today_obj + timedelta(days=idx+1)).strftime('%Y-%m-%d')
        prediction_sub_dict[prediction_date]= prediction

    predictions_dict["PredictedData"] = prediction_sub_dict

    # Assemble historical flows
    for idx in range(DAYS_TO_GO_BACK,0, -1):
        historical_date = (date_today_obj - timedelta(days=idx)).strftime('%Y-%m-%d')
        historical_flow = daily_kaituna_data[target_variable].loc[historical_date]
        historical_sub_dict[historical_date] = historical_flow

    historical_sub_dict["Today"] = daily_kaituna_data.loc[date_today][target_variable]
    predictions_dict["HistoricalData"] = historical_sub_dict

    # Convert Dictionary to JSON String
    data_string = json.dumps(predictions_dict, indent=2, default=str)

    # Upload JSON String to an S3 bucket
    # Upload to bucket #todo: Need to change this to however lambda function will do it.
    with open("secrets.json") as f:
        secrets = json.load(f)
    #Creating Session With Boto3.
    session = boto3.Session(
    aws_access_key_id=secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=secrets["AWS_ACCESS_SECRET_KEY"]
    )

    #Creating S3 Resource From the Session.
    s3 = session.resource('s3')
    object = s3.Object(secrets["BUCKET_NAME"], 'data.json')
    object.put(Body=data_string)

if __name__=="__main__":
    handle_predictions(1, 2)


