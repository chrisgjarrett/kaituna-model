# syntax=docker/dockerfile:1
FROM python:3.9-slim
#FROM tiangolo/uvicorn-gunicorn-starlette:python3.9
WORKDIR /kaituna-predictor
COPY ./requirements.txt /var/www/requirements.txt
RUN pip install -r /var/www/requirements.txt
COPY  model /kaituna-predictor/model
COPY data_preprocessing /kaituna-predictor/data_preprocessing
COPY web_scraper /kaituna-predictor/web_scraper
COPY app.py /kaituna-predictor
ENV FLASK_APP=app.py
CMD ["flask","run", "--host=0.0.0.0"]
