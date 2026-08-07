import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb


st.set_page_config(page_title="Codeless ML Interpreter", layout="wide")
st.title("Codeless ML Interpreter")
st.caption("Train a classifier, inspect its performance, and explain its behavior.")


def load_data(file):
    data = pd.read_csv(file)
    data.columns = [str(column).strip().replace(" ", "_") for column in data.columns]
    return data


def builtin_data(name):
    data = {
        "Iris": datasets.load_iris(),
        "Wine": datasets.load_wine(),
        "Breast Cancer": datasets.load_breast_cancer(),
    }[name]
    frame = pd.DataFrame(data.data, columns=data.feature_names)
    frame["target"] = data.target
    return frame


def prepare_data(data, target_column):
    data = data.dropna(subset=[target_column]).copy()
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(data[target_column])
    features = data.drop(columns=[target_column]).copy()
    if features.empty:
        raise ValueError("Choose a target while leaving at least one feature column.")

    for column in features.columns:
        if pd.api.types.is_numeric_dtype(features[column]):
            features[column] = features[column].fillna(features[column].median())
        else:
            features[column] = features[column].fillna("Missing").astype(str)
            features[column] = LabelEncoder().fit_transform(features[column])

    if len(np.unique(y)) < 2:
        raise ValueError("The target column must contain at least two classes.")
    return features, y, target_encoder


def make_model(name):
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=1)
    if name == "XGBoost":
        return xgb.XGBClassifier(eval_metric="mlogloss", n_jobs=1, random_state=42)
    if name == "LightGBM":
        return lgb.LGBMClassifier(n_jobs=1, random_state=42, verbosity=-1)
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=42)
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, n_jobs=1, random_state=42)
    return SVC(probability=True, random_state=42)


@st.cache_resource(show_spinner=False)
def train_and_analyze(features, labels, model_name):
    class_counts = np.bincount(labels)
    stratify = labels if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=stratify
    )
    model = make_model(model_name)
    model.fit(X_train, y_train)
    importance = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1
    )
    return model, X_train, X_test, y_train, y_test, importance.importances_mean


def probability_for_class(model, row, class_index):
    if not hasattr(model, "predict_proba"):
        return None
    probabilities = model.predict_proba(row.to_frame().T)[0]
    return probabilities[class_index]


dataset_option = st.sidebar.selectbox(
    "Choose a dataset", ["Select", "Upload CSV", "Iris", "Wine", "Breast Cancer"]
)
dataframe = None
if dataset_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            dataframe = load_data(uploaded_file)
        except (UnicodeDecodeError, pd.errors.ParserError) as error:
            st.error(f"Could not read this CSV: {error}")
elif dataset_option != "Select":
    dataframe = builtin_data(dataset_option)

if dataframe is None:
    st.info("Choose a built-in dataset or upload a CSV to begin.")
    st.stop()

st.subheader("Data Preview")
st.dataframe(dataframe.head(20), use_container_width=True)
default_target_index = (
    dataframe.columns.get_loc("target") if "target" in dataframe.columns else len(dataframe.columns) - 1
)
target_column = st.sidebar.selectbox(
    "Select target variable", dataframe.columns, index=default_target_index
)
model_name = st.sidebar.selectbox(
    "Select model",
    ["Random Forest", "XGBoost", "LightGBM", "Gradient Boosting", "Logistic Regression", "Support Vector Machine"],
)

try:
    X, y, target_encoder = prepare_data(dataframe, target_column)
    if len(X) < 5:
        raise ValueError("Provide at least five usable rows.")
    with st.spinner("Training model and calculating feature importance..."):
        model, X_train, X_test, y_train, y_test, importances = train_and_analyze(
            X, y, model_name
        )
except (ValueError, TypeError) as error:
    st.error(f"This dataset cannot be trained with the selected settings: {error}")
    st.stop()

y_pred = model.predict(X_test)
st.subheader("Model Performance")
left, right = st.columns(2)
with left:
    st.metric("Test accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
    st.text(classification_report(y_test, y_pred, zero_division=0))
with right:
    figure, axis = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=axis)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    st.pyplot(figure, clear_figure=True)

st.subheader("Global Feature Importance")
importance_frame = pd.DataFrame(
    {"feature": X.columns, "importance": importances}
).sort_values("importance", ascending=False)
st.bar_chart(importance_frame.set_index("feature"))
st.caption("Importance is the average drop in test accuracy when a feature is shuffled.")

st.subheader("Local Prediction Explanation")
row_position = st.selectbox("Test record to inspect", range(len(X_test)), format_func=lambda item: f"Record {item + 1}")
row = X_test.iloc[row_position].copy()
predicted_class = int(y_pred[row_position])
predicted_label = target_encoder.inverse_transform([predicted_class])[0]
actual_label = target_encoder.inverse_transform([int(y_test[row_position])])[0]
prediction_text = f"Predicted class: {predicted_label} | Actual class: {actual_label}"
st.write(prediction_text)

if hasattr(model, "predict_proba"):
    base_probability = probability_for_class(model, row, predicted_class)
    effects = []
    for feature in X.columns:
        altered_row = row.copy()
        altered_row[feature] = X_train[feature].median()
        altered_probability = probability_for_class(model, altered_row, predicted_class)
        effects.append({"feature": feature, "probability_change": base_probability - altered_probability})
    effect_frame = pd.DataFrame(effects).sort_values("probability_change")
    st.bar_chart(effect_frame.set_index("feature"))
    st.caption("Each bar shows how the predicted-class probability changes when that feature is replaced by its training median.")
else:
    st.info("The selected model does not provide probabilities for a local explanation.")

st.subheader("Misclassified Records")
misclassified = X_test.copy()
misclassified["actual"] = target_encoder.inverse_transform(y_test)
misclassified["predicted"] = target_encoder.inverse_transform(y_pred)
misclassified = misclassified[misclassified["actual"] != misclassified["predicted"]]
if misclassified.empty:
    st.success("No test records were misclassified.")
else:
    st.dataframe(misclassified, use_container_width=True)
