#!/bin/bash
app="kaituna.predictor"
docker build -t ${app} .
docker run -d -p 8080 --name=${app} -v "$PWD" ${app}
