FROM tiangolo/uwsgi-nginx-flask:python:3.9-slim
#FROM tiangolo/uvicorn-gunicorn-starlette:python3.9
RUN apt update && \
apt-get install -y \
bash nano
ENV STATIC_URL /static
ENV STATIC_PATH /var/www/app/static
COPY ./requirements.txt /var/www/requirements.txt
RUN pip install -r /var/www/requirements.txt
