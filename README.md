## Detecting NLP Data Drift using SageMaker Custom Model Monitor

This Example is an extension of Fine-tuning a PyTorch BERT model and deploying it with Amazon Elastic Inference on Amazon SageMaker aws blog [post](https://aws.amazon.com/blogs/machine-learning/fine-tuning-a-pytorch-bert-model-and-deploying-it-with-amazon-elastic-inference-on-amazon-sagemaker/). We will use the dataset and model thats outlined in this blog and extend it to demo custom model monitoring capability using [SageMaker Model Monitor](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html)

Detecting data drift in NLP is a challenging task. Model monitoring becomes an important aspect in MLOPs because the change in data distribution during inference time can cause Model decay. ML models are probabilistic and trained on certain corpus of historical data. Drift is distribution change between the training and deployment data, which is concerning if model performance changes.

We will begin with creating PyTorch Model using previously trained model artifacts. We will deploy the model to a SageMaker real time endpoint. To establish a baseline of training data distribution we will calculate BERT sentence embedding and use that in the custom model monitoring scripts to compare the real time inference traffic to compare a distance metrics to determine the deviation from training distribution

## How to execute the notebook?

Clone the repositiory and run the notebook `detect-data-drift-nlp-custom-model-monitor.ipynb` with notebook kernel set to `conda_pytorch_p36` or `Data Science Kernel` in case of running it on SageMaker Studio 

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

