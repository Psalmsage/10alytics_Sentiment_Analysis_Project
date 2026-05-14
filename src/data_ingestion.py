import pandas as pd
import numpy as np

import logging

logging.basicConfig(
    level=logging.DEBUG, format= "%(asctime)s - %(levelname)s - %(message)s")

from config.constant import input_data

def data_ingestion():
    try:
        data=pd.read_csv(input_data)
        logging.info(f"Data Ingestion is completed......")
        print(data.head())
        return data
    except Exception as e:
        logging.error(f"Error occurred during data ingestion: {e}")
        
data_ingestion()