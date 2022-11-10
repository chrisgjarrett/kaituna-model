#!/bin/bash
app="kaituna.predictor"
docker rm ${app}
docker build -t ${app} .
docker run -d -p 80:8080 ${app}
