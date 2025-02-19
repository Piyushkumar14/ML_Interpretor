import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn import datasets

# Set Streamlit page config
st.set_page_config(page_title="Codeless ML Interpreter", layout="wide")
st.title("Codeless ML Interpreter")
st.markdown("Interpret and analyze black-box ML models with ease.")

# File Upload or Inbuilt Dataset Selection
data_option = st.sidebar.selectbox("Choose a dataset", ["Select", "Upload CSV", "Iris", "Wine", "Breast Cancer"])

df = None  # Initialize df to avoid reference errors


def load_data(file):
    df = pd.read_csv(file)
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df


if data_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
elif data_option in ["Iris", "Wine", "Breast Cancer"]:
    dataset_dict = {
        "Iris": datasets.load_iris(),
        "Wine": datasets.load_wine(),
        "Breast Cancer": datasets.load_breast_cancer()
    }
    data = dataset_dict[data_option]
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    st.sidebar.success(f"Using inbuilt dataset: {data_option}")

# Process dataset
if df is not None:
    target_col = st.sidebar.selectbox("Select target variable", df.columns)
    X = df.drop(columns=[target_col])
    y = LabelEncoder().fit_transform(df[target_col])

    # Check if all classes have at least 2 samples for stratification
    unique_counts = np.bincount(y)
    if np.min(unique_counts) < 2:
        st.warning("Some classes have only one sample. Stratified split is disabled.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Model Selection
    model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost", "LightGBM", "Gradient Boosting",
                                                         "Logistic Regression", "Support Vector Machine"])

    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_choice == "XGBoost":
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
    elif model_choice == "LightGBM":
        model = lgb.LGBMClassifier(n_jobs=-1)
    elif model_choice == "Gradient Boosting":
        model = GradientBoostingClassifier()
    elif model_choice == "Logistic Regression":
        model = LogisticRegression(n_jobs=-1)
    elif model_choice == "Support Vector Machine":
        model = SVC()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model Performance
    st.subheader("Model Performance")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.sidebar.info(
        "This tool helps analyze black-box models by providing feature importance insights and model evaluation.")
else:
    st.warning("Please select a dataset to proceed.")
