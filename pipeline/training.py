from config.constant import model_name, training_args
from src.data_preprocessing import Prepare_sentiment_data
from src.model_pusher import ModelPusher
from src.model_training import ModelTraining

import logging
import os
logging.basicConfig(
    level=logging.DEBUG, format= "%(asctime)s-%(levelname)s-%(message)s)"
)



def Train_model():
    try:
        train_dataset, test_dataset = Prepare_sentiment_data()
        model= ModelTraining()
        trainer=model.model_training(train_dataset = train_dataset, test_dataset = test_dataset)
        result = model.model_evaluation(trainer)
        print(result)
        pusher = ModelPusher()
        pusher.updated_model_pusher(trainer, result)
        return trainer, result
    except Exception as e:
        logging.error(f"Error occurred during the training process: {e}")
        