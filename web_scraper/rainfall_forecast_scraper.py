import requests
import pandas as pd

# URL to scrape for forecast
data_url = "https://www.yr.no/api/v0/locations/2-2183265/forecast"

def get_rain_forecast(length_of_prediction):

    # All data
    content = requests.get(data_url).json()

    # Daily data
    daily_data = content["dayIntervals"]

    daily_data_norm = pd.json_normalize(daily_data, max_level=1)
    relevant_data_df = daily_data_norm.loc[1:length_of_prediction, ["start", "precipitation.value"]]

    # Format and configure dataframe
    relevant_data_df["start"] = pd.to_datetime(relevant_data_df['start'])
    relevant_data_df = relevant_data_df.rename(columns={
        "start":"Date",
        "precipitation.value":"Precipitation"})

    return relevant_data_df

if __name__=="__main__":
    print(get_rain_forecast(3))