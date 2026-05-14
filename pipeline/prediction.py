from utils.model_utils import load_registered_model
from src.data_cleaning import DataCleaning

import logging
logging.basicConfig(
    level=logging.DEBUG, format= "%(asctime)s-%(levelname)s-%(message)s)"
)

class predict_sentiment:
    def __init__(self):
        self.pipeline = load_registered_model()
        #define the label
        self.id2label = {0:"negative",1:"neutral",2:"positive"}
    
    def predict(self,text):
        raw_results = self.pipeline(text)
        for item in raw_results:
            index=int(item["label"].split("_")[-1])
            item['label'] = self.id2label.get(index, item["label"])
        return raw_results