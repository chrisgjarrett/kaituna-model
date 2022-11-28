from kaituna_common.web_scraper import kaituna_web_scraper
from datetime import datetime
import pickle as pk
import json

def lambda_handler(event, context):
    """This gets deployed to AWS to make predictions and update the json data in the bucket"""

    # columns needed: rainfall 3 day's forecast, today's flow, today's rainfall, today's lakelevel

    # Scrape data
    date_today = datetime.today().strftime('%Y-%m-%d')
    hourly_data = kaituna_web_scraper.collate_kaituna_data(date_today, date_today)
    print(hourly_data.head())
    
    # Preprocess
    preprocessed_data = pk.load(open('preprocessing/preprocessor.pkl', 'rb'))

    # Predict
    model = pk.load(open('model_files/model.pkl', 'rb'))

    #model.predict()

    # Assemble to json file

    # Upload to bucket

    print("hello")
    return




