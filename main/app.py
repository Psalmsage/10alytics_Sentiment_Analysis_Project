import logging
from fastapi import FastAPI,UploadFile,File
import pandas as pd
import numpy as np
from pydantic import BaseModel
from pipeline.training import Train_model
from pipeline.prediction import predict_sentiment

logging.basicConfig(
    level=logging.DEBUG, format= "%(asctime)s-%(levelname)s-%(message)s)"
)

app = FastAPI(title= "SHOPEASE Sentiment Analysis API")

class TextRequest(BaseModel):
    text: str
    
#load the model at the start of the application as the API startup
predictor = predict_sentiment()
logging.info("Model loaded successfully at startup")
app.post("/predict")

def predict_user_sentiment(request: TextRequest):
    try:
        logging.info(f"Received prediction request for text: {request.text}")
        results = predictor.predict(request.text)
        top_label = max(results, key=lambda x: x["score"])
        return[
            {
                "label": top_label["label"],
                "confidence": float(top_label["score"])
            }
        ]
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return {"error": {str(e)}}

app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    content= await file.read()
    df=pd.read_csv(io.stringIO(content.decode("utf-8")))
    
    #check if the required column(review) exists
    if "review" not in df.columns:
        logging.error("Missing 'review' column in the uploaded file")
        return {"error": "Missing 'review' column in the uploaded csv_file"}
    #predict the sentiment for each review
    result_list=[]
    for idx,row in df.iterrows():
        try:
            review = str(row["review"])
            #call the predict function for each review
            result = predictor.predict(review)
            if result is None or len(result) == 0:
                raise ValueError("Prediction returned empty result")
            top_label=max(result, key=lambda x: x["score"])
            result_row = row.to_dict()
            result_row["sentiment_label"]= top_label["label"]
            result_row["sentiment_confidence"]= float(top_label["score"])
            result_list.append(result_row)
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logging.error(f"error for review{idx}:{error_details}")
            result_row = row.to_dict()
            result_row["sentiment_label"]= "error"
            result_row["sentiment_confidence"]= 0.0
            result_list.append(result_row)
    return result_list
        
         
        
app.get("/train")
def train_model():
    try:
        Train_model()
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        return {"error": str(e)}
    
