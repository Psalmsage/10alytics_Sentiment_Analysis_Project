import mlflow
from mlflow.tracking import MlflowClient
import os
import dagshub
import logging
from config.constant import registered_model_name,model_name

def get_best_model(experiment_name = "10alytic_sentiment_analysis"):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    
    runs= client.search_runs([experiment.experiment_id])
    if not runs:
        return None
    
    best_model = sorted(
        runs,
        key=lambda x:x.data.metrics.get("f1", 0),
        reverse=True
    )[0]
    
    return best_model

def get_best_f1(experiment_name = "10alytic_sentiment_analysis"):
    best_model = get_best_model(experiment_name)
    if best_model is None:
        return None
    return best_model.data.metrics.get("f1", 0)

def load_registered_model(model_name= registered_model_name):
    dagshub.init(repo_owner='olonijayesamson',
                 repo_name='10alytics_Sentiment_Analysis_Project',
                 mlflow=True)
    model_uri = f"models:/{model_name}/latest"
    sentiment_pipeline = mlflow.transformers.load_model(model_uri)
    return sentiment_pipeline