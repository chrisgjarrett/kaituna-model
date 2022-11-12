from datetime import datetime, timedelta, date

import pandas as pd

from web_scraper import kaituna_web_scraper
from data_preprocessing import data_aggregation
import pickle as pk
import tensorflow as tf

# web frameworks
from flask import Flask

# NUMBER OF DAYS TO LOOK BACK
NUMBER_OF_DAYS_LAG = 0


# Preprocess data
def preprocess_data(X_raw):
    f = open("data_preprocessing/preprocessor.pkl", "rb")
    preprocessor = pk.load(f)
    f.close()
    return preprocessor.transform(X_raw)


# Takes a start date and end date and returns the features
def get_features():
    # Predictions require today's data only, currently
    start_date = date.today()
    end_date = start_date - timedelta(days=NUMBER_OF_DAYS_LAG)

    hourly_df = kaituna_web_scraper.collate_kaituna_data(start_date=start_date, end_date=end_date)

    # Aggregate to daily metrics
    X_daily = data_aggregation.aggregate_to_daily(hourly_df)
    X_daily["IsWeekend"] = X_daily.apply(data_aggregation.is_weekend, axis=1)  # todo: this is probably wrong

    # Get relevant features
    X_daily = X_daily[["FlowRate", "Rainfall", "LakeLevel", "IsWeekend"]]

    # Rename columns
    X_daily.columns = ["Target_lag", "Rainfall_lag", "LakeLevel_lag", "IsWeekend"]

    return X_daily


app = Flask(__name__)


@app.route("/predict-flow", methods=["GET"])
def predict_flow():
    # Get raw features
    X = get_features()

    # Preprocess
    X_preprocessed = pd.DataFrame(preprocess_data(X), columns=X.columns, index=X.index)

    # Give to model for prediction
    model = tf.keras.models.load_model('model/saved_model.h5')
    y_preds = model.predict(X_preprocessed)

    return """
        <html>
            <body>
                <p> %s: <b> %s </b> </p>
                <p> %s: <b> %s </b> </p>
                <p> %s: <b> %s </b> </p>
            </body>
        </figure>
        </html>
        """ % (date.today(), y_preds[0, 0],
               date.today() + timedelta(days=1), y_preds[0, 1],
               date.today() + timedelta(days=2), y_preds[0, 2])


@app.route("/")
def form():
    return """
        <h1> Kaituna Flow Prediction </h1>
        <p> Let's predict the flow </p>
        <br>
        <form action = "/predict-flow" method="get">
            <button name="btn">Refresh</button>
        </form>
        """


"""@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")
"""

if __name__ == "__main__":
    # port = int(os.environ.get("PORT", 8008))
    # uvicorn.run(app)
    app.run()
