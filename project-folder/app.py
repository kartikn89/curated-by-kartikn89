import streamlit as st
import pandas as pd
import joblib

st.title("Adult Census Income Classification")

model_choice = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Random Forest", "XGBoost"]
)

model_map = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

model = joblib.load(model_map[model_choice])

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    predictions = model.predict(df)
    st.write("Predictions:")
    st.write(predictions)