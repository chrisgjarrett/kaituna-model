from zipfile import ZipFile

def zip_files_for_deployment():
    """Zips the files needed to deploy model"""
    # create a ZipFile object
    zipObj = ZipFile('handle_predictions.zip', 'w')

    # Add multiple files to the zip # Todo can these strings be replaced programmitically somehow?
    zipObj.write('handle_predictions.py')
    zipObj.write('kaituna_common/web_scraper/kaituna_web_scraper.py')
    zipObj.write('kaituna_common/web_scraper/rainfall_forecast_scraper.py')
    zipObj.write('preprocessing/aggregate_hourly_data.py')
    zipObj.write('preprocessing/preprocessor.pkl')
    zipObj.write('model_files/model.pkl')

    # close the Zip File
    zipObj.close()

if __name__=="__main__":
    zip_files_for_deployment()