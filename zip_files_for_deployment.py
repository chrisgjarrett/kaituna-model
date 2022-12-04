from tarfile import TarFile, TarInfo
 
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
    tar = TarFile('handle_predictions.tar.gz', 'w')

    # Add multiple files to the zip # Todo can these strings be replaced programmitically somehow?
    for file_name in files_to_zip:

        # Set global read permissions
        info = TarInfo(file_name)
        info.mode = 0o100755 << 16
        tar.add(file_name)    

    # close the Zip File
    tar.close()

if __name__=="__main__":
    zip_files_for_deployment()