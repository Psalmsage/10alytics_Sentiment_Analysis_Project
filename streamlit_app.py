import streamlit as st 
import pandas as pd
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="SHOPEASE Sentiment Analysis Dashboard", layout="wide")
st.title("SHOPEASE Sentiment Analysis Dashboard")

st.header("Single Review Sentiment Prediction")

#add a unique key
user_input = st.text_area("Enter customer review ", key="single_review_input")
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review to predict sentiment.")
    else:
        with st.spinner("Analyzing sentiment"):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": user_input}
                )
                if response.status_code == 200:
                    result = response.json()
                    col1, col2 = st.columns(2)
                    col1.metric("Sentiment",result["label"])
                    col2.metric("Confidence", f"{result['confidence']:.2f}")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"An error occurred:{e}")
                
st.divider()

st.header("Batch Review Sentiment Prediction(csv upload)")
uploaded_file = st.file_uploader("Upload a CSV file with customer reviews column", type=["csv"],key="batch_file")
if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    st.write("Preview of the uploaded file:")
    st.dataframe(df.head())
    
    if st.button("Run Batch Sentiment",key= "batch_predict"):
        with st.spinner("Processing batch predictions........"):
            try:
                response = requests.post(
                    f"{API_URL}/predict/batch",
                    files={"file": (uploaded_file.name, uploaded_file.getValue(), "text/csv")}
                )
                
                if response.status_code == 200:
                    results =pd.DataFrame (response.json())
                    st.success("Batch prediction completed successfully!")
                    st.dataframe(results)
                    
                    csv = results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name="Sentiment_prediction_results.csv",
                        mime="text/csv",
                        key="download_results"
                    )
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"error: {e}")

st.divider()
st.header("Model retraining section")

st.warning("Note: This may take time")
if st.button("retrain Model"):
    try:
        response = requests.get(f"{API_URL}/train")
        if response.status_code == 200:
            st.success("Model triggered successfully!")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"An error occurred: {e}")