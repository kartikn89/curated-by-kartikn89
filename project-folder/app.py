import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

st.set_page_config(page_title="Adult Census Income Classifier", layout="wide")

st.title("Adult Census Income Classification")

st.markdown("""
This app predicts whether an individual's income exceeds **$50K/year**
using pre-trained machine learning models.
""")

# -------------------------------
# Model selection
# -------------------------------
model_name = st.selectbox(
    "Select a trained model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

model_paths = {
    "Logistic Regression": os.path.join(MODEL_DIR, "logistic_regression.pkl"),
    "Decision Tree": os.path.join(MODEL_DIR, "decision_tree.pkl"),
    "KNN": os.path.join(MODEL_DIR, "knn.pkl"),
    "Naive Bayes": os.path.join(MODEL_DIR, "naive_bayes.pkl"),
    "Random Forest": os.path.join(MODEL_DIR, "random_forest.pkl"),
    "XGBoost": os.path.join(MODEL_DIR, "xgboost.pkl")
}

model = joblib.load(model_paths[model_name])

# -------------------------------
# Dataset upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload test dataset (CSV only, small size recommended)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Target handling
    # -------------------------------
    if "income" not in df.columns:
        st.error("Uploaded dataset must contain the 'income' column for evaluation.")
        st.stop()

    X = df.drop(columns=["income"])
    y_true = df["income"]

    # Encode target
    y_true = y_true.map({"<=50K": 0, ">50K": 1})

    # -------------------------------
    # Predictions
    # -------------------------------
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        y_proba = None

    # -------------------------------
    # Metrics computation
    # -------------------------------
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC", "MCC"],
        "Value": [accuracy, precision, recall, f1, auc, mcc]
    })

    st.subheader("Evaluation Metrics")
    st.table(metrics_df.round(4))

    # -------------------------------
    # Confusion Matrix
    # -------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["<=50K", ">50K"],
        yticklabels=["<=50K", ">50K"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)
