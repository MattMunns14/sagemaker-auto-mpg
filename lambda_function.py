import json
import os
import io
import boto3
import csv


ENDPOINT_NAME=os.environ['ENDPOINT_NAME']
content_type = os.environ['CONTENT_TYPE']
runtime = boto3.Session().client(service_name='sagemaker-runtime',region_name='us-east-2')

def lambda_handler(event, context):
    data = json.dumps(event)

    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, 
    ContentType=content_type, 
    Body=data)
    response = json.loads(response['Body'].read().decode())
    predicted_mpg = response['predictions'][0]['score']
    return {
        'statusCode': 200,
        'body': predicted_mpg
    }
