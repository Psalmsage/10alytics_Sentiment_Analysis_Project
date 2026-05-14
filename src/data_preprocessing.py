import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import logging
import torch # type: ignore
import os
from config.constant import input_data, model_name,train_data, test_data
from src.data_cleaning import clean_data
from src.data_ingestion import data_ingestion
logging.basicConfig(
    level=logging.DEBUG, format= "%(asctime)s - %(levelname)s - %(message)s")

class data_processor:
    #Reading the dataset using the data_ingestion function and then applying the clean_data function to clean the dataset and save it to a csv file.
    def __init__(self):
        self.data = data_ingestion()
        self.clean_data = clean_data(self.data)
        
    def split_data(self):
        try:
            #x is the feature and y is the target variable
            X = self.clean_data["final_text"].astype(str)
            y = self.clean_data["label"]
            #splitting the data into training and testing sets using train_test_split function from sklearn library
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(X_train.head())
            print(X_test.head())
            logging.info(f"Data splitting is completed......")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error occurred during data splitting: {e}")
            
class tokenizerwrapper:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def encode(self,text):
        return self.tokenizer(text.to_list(), truncation=True, padding=True, max_length=128)

class SentimentDataset(torch.utils.data.Dataset):
  def __init__(self,encodings,labels):
    self.encodings = encodings

    #ensuring labels are in list to avoid any pandas list issue
    if hasattr(labels,"tolist"):
      self.labels = labels.tolist()
    elif hasattr(labels,"_iter_") and not isinstance(labels,(list,tuple)):
      self.labels = list(labels)
    else:
      self.labels = labels

  def __len__(self):
    return len(self.labels)

  def __getitem__ (self,idx):
    item = {key:torch.tensor(val[idx]) for key,val in self.encodings.items()}
    item["labels"] = torch.tensor(self.labels[idx])
    return item
    
def Prepare_sentiment_data():
    try:
        processor = data_processor()
        X_train, X_test, y_train, y_test = processor.split_data()
        tokenizer = tokenizerwrapper()
        train_encodings = tokenizer.encode(X_train)
        test_encodings = tokenizer.encode(X_test)
        
        train_dataset = SentimentDataset(train_encodings,y_train)
        test_dataset = SentimentDataset(test_encodings,y_test)
        os.makedirs(os.path.dirname(train_data), exist_ok=True)
        os.makedirs(os.path.dirname(test_data), exist_ok=True)
        torch.save(train_dataset, train_data)
        torch.save(test_dataset, test_data)
        logging.info(f"Data preparation is completed and ready to pass into the model......")
        return train_dataset, test_dataset
    except Exception as e:
        logging.error(f"Error occurred during data preparation: {e}")

Prepare_sentiment_data()
