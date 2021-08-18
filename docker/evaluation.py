"""Custom Model Monitoring script for Detecting Data Drift in NLP using SageMaker Model Monitor
"""

# Python Built-Ins:
from collections import defaultdict
import datetime
import json
import os
import traceback
from types import SimpleNamespace

# External Dependencies:
import numpy as np
import boto3
from scipy.spatial.distance import cosine
from transformers import BertTokenizer, BertModel
import torch


def get_environment():
    """Load configuration variables for SM Model Monitoring job

    See https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-byoc-contract-inputs.html
    """
    try:
        with open("/opt/ml/config/processingjobconfig.json", "r") as conffile:
            defaults = json.loads(conffile.read())["Environment"]
    except Exception as e:
        traceback.print_exc()
        print("Unable to read environment vars from SM processing config file")
        defaults = {}

    return SimpleNamespace(
        dataset_format=os.environ.get("dataset_format", defaults.get("dataset_format")),
        dataset_source=os.environ.get(
            "dataset_source",
            defaults.get("dataset_source", "/opt/ml/processing/input/endpoint"),
        ),
        end_time=os.environ.get("end_time", defaults.get("end_time")),
        output_path=os.environ.get(
            "output_path",
            defaults.get("output_path", "/opt/ml/processing/resultdata"),
        ),
        publish_cloudwatch_metrics=os.environ.get(
            "publish_cloudwatch_metrics",
            defaults.get("publish_cloudwatch_metrics", "Enabled"),
        ),
        sagemaker_endpoint_name=os.environ.get(
            "sagemaker_endpoint_name",
            defaults.get("sagemaker_endpoint_name"),
        ),
        sagemaker_monitoring_schedule_name=os.environ.get(
            "sagemaker_monitoring_schedule_name",
            defaults.get("sagemaker_monitoring_schedule_name"),
        ),
        start_time=os.environ.get(
            "start_time", 
            defaults.get("start_time")),
        max_ratio_threshold=float(os.environ.get(
            "THRESHOLD", 
             defaults.get("THRESHOLD", "nan"))),
        bucket=os.environ.get(
            "bucket",
            defaults.get("bucket", "None")),
    )


def download_embeddings_file():
    
    env = get_environment()
    from s3fs.core import S3FileSystem
    s3 = S3FileSystem()
    
    key = 'sagemaker/nlp-data-drift-bert-model/embeddings/embeddings.npy'
    bucket = env.bucket
    print("S3 bucket name is",bucket)

    return np.load(s3.open('{}/{}'.format(bucket, key)))
    
if __name__=="__main__":

    env = get_environment()
    print(f"Starting evaluation with config\n{env}")

    print(f"Downloading Embedding File")
    
    #download BERT embedding file used for fine-tuning BertForSequenceClassification
    embedding_list = download_embeddings_file()
    
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states = True, # Whether the model returns all hidden-states.
                                      )

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()
    
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    sent_cosine_dict = {}
    violations = []
    
    total_record_count = 0  # Including error predictions that we can't read the response for
    error_record_count = 0
    counts = defaultdict(int)  # dict defaulting to 0 when unseen keys are requested
    for path, directories, filenames in os.walk(env.dataset_source):
        for filename in filter(lambda f: f.lower().endswith(".jsonl"), filenames):
            with open(os.path.join(path, filename), "r") as file:
                for entry in file:
                    total_record_count += 1
                    try:
                        response = json.loads(json.loads(entry)["captureData"]["endpointInput"]["data"])
                    except:
                        continue
                
                    for record in response:
                        encoded_dict = tokenizer.encode_plus(
                            record, 
                            add_special_tokens = True,
                            max_length = 64,
                            padding= True,
                            return_attention_mask = True,
                            return_tensors = 'pt',
                            truncation=True,
                            )

                        with torch.no_grad():
                            outputs = model(encoded_dict['input_ids'], encoded_dict['attention_mask'])
                            hidden_states = outputs[2]
                            token_vecs = hidden_states[-2][0]
                            input_sentence_embedding = torch.mean(token_vecs, dim=0)
                        
                        cosine_score = 0
                        
                        for embed_item in embedding_list:
                            cosine_score += (1 - cosine(input_sentence_embedding, embed_item))
                        cosine_score_avg = cosine_score/(len(embedding_list))
                        if cosine_score_avg < env.max_ratio_threshold:
                            error_record_count += 1
                            sent_cosine_dict[record] = cosine_score_avg
                            violations.append({
                                    "sentence": record,
                                    "avg_cosine_score": cosine_score_avg,
                                    "feature_name": "sent_cosine_score",
                                    "constraint_check_type": "baseline_drift_check",
                                    "endpoint_name" : env.sagemaker_endpoint_name,
                                    "monitoring_schedule_name": env.sagemaker_monitoring_schedule_name
                                })
        
    print("Checking for constraint violations...")
    print(f"Violations: {violations if len(violations) else 'None'}")

    print("Writing violations file...")
    with open(os.path.join(env.output_path, "constraints_violations.json"), "w") as outfile:
        outfile.write(json.dumps(
            { "violations": violations },
            indent=4,
        ))
    
    print("Writing overall status output...")
    with open("/opt/ml/output/message", "w") as outfile:
        if len(violations):
            msg = ''
            for v in violations:
                msg += f"CompletedWithViolations: {v['sentence']}"
                msg +="\n"
        else:
            msg = "Completed: Job completed successfully with no violations."
        outfile.write(msg)
        print(msg)

    if True:
    #if env.publish_cloudwatch_metrics:
        print("Writing CloudWatch metrics...")
        with open("/opt/ml/output/metrics/cloudwatch/cloudwatch_metrics.jsonl", "a+") as outfile:
            # One metric per line (JSONLines list of dictionaries)
            # Remember these metrics are aggregated in graphs, so we report them as statistics on our dataset
            outfile.write(json.dumps(
            { "violations": violations },
            indent=4,
            ))
    print("Done")