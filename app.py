import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pandas as pd
import numpy as np
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

st.title("Loan Approval Classification Dashboard")

# File path
file_path = r"D:\loan_data_detection\dataset\loan_data.csv"

# File Upload or Automatic Load
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    st.success("File loaded automatically from the predefined path.")
else:
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully.")

# Dataset Preview
if 'df' in locals():  # Check if df is loaded
    st.write("Dataset Preview:", df.head())
else:
    st.error("No dataset available. Please upload a file or check the predefined path.")

# Data Information
if st.checkbox("Show Data Info"):
    buffer = io.StringIO()  # Create a buffer to capture df.info() output
    df.info(buf=buffer)  # Write df.info() output to the buffer
    info_output = buffer.getvalue()  # Retrieve the buffer content as a string
    st.text(info_output)  # Display the information in Streamlit

# Descriptive Statistics
if st.checkbox("Show Descriptive Statistics"):
    st.write(df.describe())

# Null Values
if st.checkbox("Show Null Values Count"):
    st.write(df.isnull().sum())

# Fill Missing Values
df["loan_percent_income"].fillna(method="ffill", inplace=True)
df["cb_person_cred_hist_length"].fillna(method="ffill", inplace=True)
df["credit_score"].fillna(method="ffill", inplace=True)
df["previous_loan_defaults_on_file"].fillna(method="ffill", inplace=True)
df["loan_status"].fillna(method="ffill", inplace=True)

# Detect and Warn About Non-Numeric Columns
non_numeric_cols = df.select_dtypes(exclude=['number']).columns
if len(non_numeric_cols) > 0:
    st.warning(f"Non-numeric columns detected: {list(non_numeric_cols)}. They will be encoded.")

# Encoding Categorical Variables
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    if df[col].nunique() <= 10:  # Suitable for low-cardinality categorical data
        df[col] = df[col].astype('category').cat.codes
    else:
        st.warning(f"Column '{col}' has high cardinality or unexpected values. Please handle it manually.")

# Recheck for Non-Numeric Columns
non_numeric_cols = df.select_dtypes(exclude=['number']).columns
if len(non_numeric_cols) > 0:
    st.error(f"Non-numeric columns remain: {list(non_numeric_cols)}. Ensure all columns are numeric.")
else:
    st.write("Data After Encoding:", df.head())

# Correlation Heatmap
if st.checkbox("Show Correlation Heatmap"):
    numeric_df = df.select_dtypes(include=["int64", "float64"])  # Filter numeric columns
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Ensure 'loan_status' exists and is numeric
if 'loan_status' not in df.columns:
    st.error("Column 'loan_status' is missing from the dataset.")
else:
    if df['loan_status'].dtype == 'object':
        df['loan_status'] = df['loan_status'].astype('category').cat.codes

    # Correlation and Feature Selection
    numeric_df = df.select_dtypes(include=['number'])
    threshold = 0.1
    correlation_matrix = numeric_df.corr()
    high_corr_features = correlation_matrix.index[abs(correlation_matrix["loan_status"]) > threshold].tolist()

    if "loan_status" in high_corr_features:
        high_corr_features.remove("loan_status")

    X_selected = df[high_corr_features]
    Y = df["loan_status"]

    # Check for Missing Values in X_selected
    if X_selected.isnull().sum().sum() > 0:
        st.error("Feature set contains missing values. Please handle them before proceeding.")
    else:
        # Data Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)

        # Train-Test Split
        X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

        # Modeling Options
        model_option = st.selectbox("Choose a Model", ["Logistic Regression", "SVM", "K-Nearest Neighbors"])

        if model_option == "Logistic Regression":
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_option == "SVM":
            model = SVC()
        elif model_option == "K-Nearest Neighbors":
            k = st.slider("Choose the value of K", min_value=1, max_value=20, value=3)
            model = KNeighborsClassifier(n_neighbors=k)

        # Train the Model
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        # Model Evaluation
        accuracy = accuracy_score(Y_test, Y_pred)
        conf_matrix = confusion_matrix(Y_test, Y_pred)
        class_report = classification_report(Y_test, Y_pred, output_dict=True)

        st.write(f"Model Accuracy: {accuracy:.4f}")
        st.write("Confusion Matrix:")
        st.write(conf_matrix)

        st.write("Classification Report:")
        st.json(class_report)

        # Confusion Matrix Heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Predicted Negative", "Predicted Positive"],
                    yticklabels=["Actual Negative", "Actual Positive"], ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix Heatmap")
        st.pyplot(fig)

        # Approve Loan or Not
        st.write("### Approve Loan Based on Model Prediction")
        sample_input = st.text_input(f"Enter {X_selected.shape[1]} feature values (comma-separated)", "")
        if sample_input:
            try:
                sample_values = [float(x) for x in sample_input.split(",")]
                if len(sample_values) != X_selected.shape[1]:
                    st.error(f"Input should have exactly {X_selected.shape[1]} values.")
                else:
                    sample_scaled = scaler.transform([sample_values])
                    prediction = model.predict(sample_scaled)
                    st.write("Loan Status Prediction:", "Approved" if prediction[0] == 1 else "Rejected")
            except ValueError as e:
                st.error(f"Input contains invalid characters. Ensure only numeric values are provided. Error: {e}")
