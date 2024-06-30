import os
from dotenv import load_dotenv
# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# ex_model_name_or_path = os.path.join(script_dir, '..', 'fine-tuned-extractive')

from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
import boto3
import torch
import io
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')

def list_files_in_s3_directory(bucket_name, prefix):
    s3 = boto3.client('s3',aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                      region_name=AWS_DEFAULT_REGION)
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
    files = []
    for page in pages:
        for obj in page.get('Contents', []):
            files.append(obj['Key'])
    return files

def load_model_files_from_s3(bucket_name, prefix):
    s3 = boto3.client('s3',aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                      region_name='eu-north-1')
    files = list_files_in_s3_directory(bucket_name, prefix)
    
    model_files = {}
    for file_key in files:
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        file_data = response['Body'].read()
        file_name = os.path.basename(file_key)
        model_files[file_name] = file_data
    
    return model_files
# Example usage
bucket_name = 'modelfinetuned'
prefix = 'fine-tuned-abstractive/'  # Directory in S3


model=load_model_files_from_s3(bucket_name,prefix)
ex_local_tokenizer=AutoTokenizer.from_pretrained(model)
ex_local_model=AutoModelForSeq2SeqLM.from_pretrained(model)


# ab_model_name_or_path=os.path.join(script_dir, '..', 'fine-tuned-abstractive')
ab_local_tokenizer=AutoTokenizer.from_pretrained(model)
ab_local_model=AutoModelForSeq2SeqLM.from_pretrained(model)


from transformers import pipeline


def Extract(article:str,max_length=200):
    ex_summarize=pipeline('summarization',model=ex_local_model,tokenizer=ex_local_tokenizer,max_length=max_length)
    return ex_summarize(article)[0]['summary_text']
def Abstract(article:str,max_length=200):
    ab_summarize=pipeline('summarization',model=ab_local_model,tokenizer=ab_local_tokenizer,max_length=max_length)
    return ab_summarize(article)[0]['summary_text']