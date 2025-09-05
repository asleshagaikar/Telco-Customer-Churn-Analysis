import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from io import BytesIO

# Load trained model and processed files
model = pickle.load(open("rf_model.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))
feature_importances = pickle.load(open("feature_importances.pkl", "rb"))
df_encoded = pickle.load(open("df_dummies.pkl", "rb"))  # Preprocessed dataset
scaler = pickle.load(open("scaler.pkl", "rb")) 

# Define file path for existing template
TEMPLATE_FILE_PATH = "User_Input.xlsx"

# Load original dataset for user-friendly input
file_path = "Customer_Churn_dataset.csv"
df_raw = pd.read_csv(file_path)

# Sidebar navigation
st.sidebar.title("Churn Prediction Dashboard")
option = st.sidebar.radio("Select an option", ["Exploratory Data Analysis", "Churn Prediction"])

# ---------------------------------------
# Exploratory Data Analysis (EDA)
# ---------------------------------------
if option == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")

    # Churn Distribution
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=df_raw["Churn"], palette="Set2", ax=ax)
    st.pyplot(fig)

    # Tenure vs. Churn
    st.subheader("Tenure vs. Churn")
    fig, ax = plt.subplots()
    sns.boxplot(x="Churn", y="tenure", data=df_raw, ax=ax)
    st.pyplot(fig)

    # Contract Type vs. Churn
    st.subheader("Contract Type vs. Churn")
    contract_churn = df_raw.groupby(["Contract", "Churn"]).size().unstack()
    fig, ax = plt.subplots()
    (contract_churn.T * 100.0 / contract_churn.T.sum()).T.plot(
        kind="bar", stacked=True, width=0.3, figsize=(8, 5), ax=ax
    )
    st.pyplot(fig)

    # MonthlyCharges vs. Churn
    st.subheader("Monthly Charges vs. Churn")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="Churn", y="MonthlyCharges", data=df_raw, ax=ax)
    st.pyplot(fig)

    # TotalCharges vs. Churn
    st.subheader("Total Charges vs. Churn")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="Churn", y="TotalCharges", data=df_raw, ax=ax)
    st.pyplot(fig)

    # Feature Importance
    st.subheader("Feature Importance")
    feature_data = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
    feature_data_sorted = feature_data.sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.barplot(x="Importance", y="Feature", data=feature_data_sorted, ax=ax)
    st.pyplot(fig)

# ---------------------------------------
# Churn Prediction
# ---------------------------------------
elif option == "Churn Prediction":
    st.title("Customer Churn Prediction")
    st.write("Download the Excel Template with customer details")

    # Download existing Excel template
    st.subheader("Download Input Template")
    with open(TEMPLATE_FILE_PATH, "rb") as file:
        st.download_button(label="Download Excel Template", data=file, file_name="Churn_Input_Template.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Upload filled user input file
    st.subheader("Upload Filled Excel File for Prediction")
    uploaded_file = st.file_uploader("Upload your filled Excel file", type=["xlsx"])

    if uploaded_file is not None:
        df_uploaded = pd.read_excel(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(df_uploaded.head())

        # One-hot encode uploaded data
        df_encoded_input = pd.get_dummies(df_uploaded)

        # Ensure the input data has the same feature columns as the trained model
        missing_cols = set(feature_names) - set(df_encoded_input.columns)
        for col in missing_cols:
            df_encoded_input[col] = 0  # Add missing columns with default value 0

        # Reorder columns to match training data
        df_encoded_input = df_encoded_input[feature_names]

        # Scale the input data
        df_scaled_input = scaler.transform(df_encoded_input)

        # Predict churn
        predictions = model.predict(df_scaled_input)
        probabilities = model.predict_proba(df_scaled_input)[:, 1] * 100

        # Add predictions to DataFrame
        df_uploaded["Churn Prediction"] = ["Yes" if pred == 1 else "No" for pred in predictions]
        df_uploaded["Churn Probability (%)"] = probabilities

        # Show results
        st.subheader("Prediction Results")
        st.dataframe(df_uploaded)

        # Save results to Excel
        output_result = BytesIO()
        with pd.ExcelWriter(output_result, engine='xlsxwriter') as writer:
            df_uploaded.to_excel(writer, index=False, sheet_name='Predictions')
        output_result.seek(0)

        # Provide download option for results
        st.download_button(label="Download Predictions", data=output_result, file_name="Churn_Predictions.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
