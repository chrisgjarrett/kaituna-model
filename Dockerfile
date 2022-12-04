FROM public.ecr.aws/lambda/python:3.8

# Install the function's dependencies using file requirements.txt
# from your project folder.

COPY requirements-predictions.txt  .
RUN  pip3 install -r requirements-predictions.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code
COPY handle_predictions.py ${LAMBDA_TASK_ROOT}

# Web scraper code
RUN mkdir -p ${LAMBDA_TASK_ROOT}/kaituna_common/web_scraper
COPY /kaituna_common/web_scraper/kaituna_web_scraper.py ${LAMBDA_TASK_ROOT}/kaituna_common/web_scraper/
COPY kaituna_common/web_scraper/rainfall_forecast_scraper.py ${LAMBDA_TASK_ROOT}/kaituna_common/web_scraper/

# Preprocessing
RUN mkdir -p ${LAMBDA_TASK_ROOT}/preprocessing
COPY preprocessing/aggregate_hourly_data.py ${LAMBDA_TASK_ROOT}/preprocessing/
COPY preprocessing/preprocessor.pkl ${LAMBDA_TASK_ROOT}/preprocessing/

# Model files
RUN mkdir -p ${LAMBDA_TASK_ROOT}/model_files
COPY model_files/model.pkl ${LAMBDA_TASK_ROOT}/model_files/

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "handle_predictions.handle_predictions" ] 