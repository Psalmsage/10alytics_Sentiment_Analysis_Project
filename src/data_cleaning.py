import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import nltk
import re
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from src.data_ingestion import data_ingestion
import sys
from config.constant import Clean_data

logging.basicConfig(
    level=logging.DEBUG, format= "%(asctime)s - %(levelname)s - %(message)s")



sentiment_data=data_ingestion()

class DataCleaning:
    def __init__(self):
        self._ensure_nltk()
    
    def _load_nlp(self) -> spacy.language.Language:
        for model in ("en_core_web_sm", "xx_ent_wiki_sm"):
            try:
                return spacy.load(model)
            except OSError:
                continue
        nlp_fallback = spacy.blank("xx")
        return nlp_fallback
    
    def _ensure_nltk(self) -> None:
        try:
            _= stopwords.words("english")
        except LookupError:
            nltk.download("stopwords")
        try:
            word_tokenize("test")
        except LookupError:
            nltk.download("punkt")

        #Newer Nltk version split tokenizer tables into 'punkt tab'
        try:
            nltk.data.find("tokenizer/punkt_tab/english")
        except LookupError:
            nltk.download("punkt_tab")
        except Exception:
            pass
        
    def clean_text(self,text: str) -> str:
        """
        Docstrings: This function is used to clean the text,
                    convert data to lowercase
                    remove url and special characters
                    remove white spaces
                    keep accented letter and characters and regular expressions
                    return clean text
        """
        text = str(text).lower()
        text = re.sub(r"[^a-zA-ZȦ-Ẓä-ẓ0-9\s]","",text)
        text = re.sub(r"/s+"," ",text).strip()
        return text
    
    def lemantize_text(self,text: str) -> str:
        """
        Docstrings: This function is used to lemantize the text
        it groups different forms of words so that they can anaylze in single items
        example: better - good,best-good,better-well,running-run,ran - run
        and return lemantized text
        """
        nlp = self._load_nlp()
        doc = nlp(text)
        return " ".join([token.lemma_ if token.lemma_ else token.text for token in doc])

    def remove_stopwords(self,text: str) -> str:
        """
        Docstrings: This function is used to remove the stopwords(e.g the,is,them,a,an)
        this help keep only meaningful words that contribute to sentiment
        """
        tokens = word_tokenize(text)
        sw = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in sw]
        return " ".join(tokens)
    
def clean_data(data: pd.DataFrame):
    """
    Docstrings: This function is used to clean the data by applying the above functions
    and return cleaned data
    """
    try:
        Cleaner=DataCleaning()
        data['clean_text'] =data['review'].apply(Cleaner.clean_text)
        data['lemman_text'] =data['clean_text'].apply(Cleaner.lemantize_text)
        data['final_text']=data['lemman_text'].apply(Cleaner.remove_stopwords)
        
        data['label']=data['rating'].apply(lambda r : 0 if r in (1,2) else (1 if r == 3 else 2))
        data = data[["review","final_text","label"]]
        data.to_csv(Clean_data)
        
        logging.info(f"Dataset Cleaning is completed......")
        
        print(data.head())
        return data
    except Exception as e:
        logging.error(f"Error occurred during data cleaning: {e}")
        

    