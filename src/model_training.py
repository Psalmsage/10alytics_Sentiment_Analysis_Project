import transformers
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from config.constant import model_name, training_args,number_of_labels
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
import logging
from src.data_preprocessing import Prepare_sentiment_data
from src.model_pusher import ModelPusher

logging.basicConfig(
    level=logging.DEBUG, format= "%(asctime)s-%(levelname)s-%(message)s)"
)

class ModelTraining:
    def __init__(self):
        self.model= AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=number_of_labels)
        
    def compute_metrics(self, pred):
        prediction = np.argmax(pred.predictions,axis=1)
        labels = pred.label_ids
        acc = accuracy_score(labels,prediction)
        f1 = f1_score(labels,prediction,average="weighted")
        return {"accuracy":acc,"f1":f1}
    
    def model_training(self,train_dataset,test_dataset):
        try:
            trainer = Trainer(
                model = self.model,
                args = training_args,
                train_dataset = train_dataset,
                eval_dataset= test_dataset,
                compute_metrics= self.compute_metrics
            )
            trainer.train()
            logging.info(f"Model training is completed......")
            return trainer
        except Exception as e:
            logging.error(f"Error occurred during model training: {e}")
            
    def model_evaluation(self,trainer):
        result= trainer.evaluate()
        print(result)
        return result
        
        
