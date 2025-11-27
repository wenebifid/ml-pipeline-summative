# src/s3_utils.py
import boto3
import os

def download_model_from_s3(bucket, key, dest_path):
    s3 = boto3.client('s3')
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    s3.download_file(bucket, key, dest_path)

def upload_model_to_s3(bucket, key, src_path):
    s3 = boto3.client('s3')
    s3.upload_file(src_path, bucket, key)
