import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, accuracy_score


# ---------------------------Load the dataset----------------------------------
file_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'  # Update this to your local file path
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# ----------------------Data Cleaning and Preprocessing------------------------
# Check for missing values
print(data.isnull().sum())

# Convert TotalCharges to numeric and handle errors
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Fill missing values
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

# Encode categorical variables
categorical_cols = data.select_dtypes(include='object').columns
for col in categorical_cols:
    if col != 'customerID':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

#Drop irrelevant columns
data = data.drop(['customerID'], axis=1)

# Display the cleaned data
print(data.info())

#----------------------Exploratory Data Analysis (EDA)--------------------------
# Churn distribution
plt.figure(figsize=(10, 6))
sns.countplot(data['Churn'], palette='Set2')
plt.title('Churn Distribution')
plt.savefig('churn_distribution.png')

#Analyze Correlation Between Numerical Features
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig('Correlation_Heatmap.png')
# plt.show()

# MonthlyCharges vs. Churn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='MonthlyCharges', data=data)
plt.title('MonthlyCharges vs. Churn')
plt.savefig('MonthlyChargesvsChurn.png')
# plt.show()


# TotalCharges vs. Churn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='TotalCharges', data=data)
plt.title('TotalCharges vs. Churn')
plt.savefig('TotalChargesvsChurn.png')
# plt.show()


#-----------------------------Predictive Modeling--------------------------------
# Define features and target
X = data.drop('Churn', axis=1)
y = data['Churn']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Calculate Mean Absolute Error (MAE) 
mae = mean_absolute_error(y_test, y_pred) 
print(f"Mean Absolute Error (MAE): {mae}")

#------------------------------Visualization of Results---------------------------------
# Feature importance
importances = model.feature_importances_
features = X.columns

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.savefig('FeatureImportance.png')
# plt.show()
