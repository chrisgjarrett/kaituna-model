#!/bin/sh
if [ -z "${AWS_LAMBDA_RUNTIME_API}" ]; then
    exec /usr/bin/aws-lambda-rie /usr/local/bin/python -m handle_predictions $1
else
    exec /usr/local/bin/python -m handle_predictions $1
fi