from zipfile import ZipFile, ZipInfo

def zip_files_for_deployment():
    """Zips the files needed to deploy model"""

    files_to_zip = [
    'handle_predictions.py',
    'kaituna_common/web_scraper/kaituna_web_scraper.py',
    'kaituna_common/web_scraper/rainfall_forecast_scraper.py',
    'preprocessing/aggregate_hourly_data.py',
    'preprocessing/preprocessor.pkl',
    'model_files/model.pkl'
    ]

    # create a ZipFile object
    zipObj = ZipFile('handle_predictions.zip', 'w')

    # Add multiple files to the zip # Todo can these strings be replaced programmitically somehow?
    for file_name in files_to_zip:

        # Set global read permissions
        info = ZipInfo(file_name)
        info.external_attr = 0o100755 << 16
        zipObj.write(file_name)    

    # close the Zip File
    zipObj.close()

if __name__=="__main__":
    zip_files_for_deployment()