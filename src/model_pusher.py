import mlflow
import mlflow.transformers
import os
import logging
import dagshub
from transformers import pipeline
from utils.model_utils import get_best_f1
from config.constant import model_name,training_args



class ModelPusher:
    def __init__(self):
        dagshub.init(repo_owner='olonijayesamson',
                     repo_name='10alytics_Sentiment_Analysis_Project',
                     mlflow=True
        )
        self.experiment_name = "10alytic_sentiment_analysis"
        mlflow.set_experiment(self.experiment_name)
        
    def updated_model_pusher(self,trainer,metrics):
        try:
            new_f1 = metrics["eval_f1"]
            old_f1 = get_best_f1(self.experiment_name)
            
            if old_f1 is None or new_f1 > old_f1:
                with mlflow.start_run():
                    #logging the metrics and parameters of the model
                    mlflow.log_metric("accuracy", metrics["eval_accuracy"])
                    mlflow.log_metric("loss",metrics["eval_loss"])
                    mlflow.log_metric("f1",new_f1)
                    mlflow.log_param("epochs",training_args.num_train_epochs)
                    mlflow.log_param("learning_rate",training_args.learning_rate)
                    mlflow.log_param("batch_size",training_args.per_device_train_batch_size)
                    
                    #creating an hugging face pipeline on how our model will be stored and  used in production
                    
                    sentiment_pipeline = pipeline(
                        task = "text-classification",
                        model = trainer.model,
                        tokenizer = model_name,
                        return_all_scores=True
                    )
                    
                    #logging the pipeline using mlflow transformers module and registering the model in mlflow model registry
                    mlflow.transformers.log_model(
                        transformers_model=sentiment_pipeline,
                        artifact_path = "model",
                        registered_model_name = "sentiment_model"
                    )
                logging.info("New model + tokenizer logged and registered into mlflow successfully.")
            else:
                logging.info("Current model did not outperform the best model. No update to mlflow.")
        except Exception as e:
            logging.error(f"Error during model pushing: {e}")