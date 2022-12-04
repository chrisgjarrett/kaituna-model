# This script gets hourly data from the kaituna and saves it to a csv
from kaituna_web_scraper import collate_kaituna_data
from datetime import timedelta, date

START_DATE = "2018-01-01"
END_DATE = date.today() - timedelta(days=1)

if __name__ == "__main__":
    
    # Get data 
    df_river_data = collate_kaituna_data(start_date=START_DATE, end_date=END_DATE)

    # Save to csv
    output_file_name = START_DATE.join(["_", START_DATE, END_DATE.strftime('%Y-%m-%d'), "hourly_kaituna_data.csv"])
    df_river_data.to_csv(output_file_name)