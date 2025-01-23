# README: Customer Churn Prediction with Random Forest Classifier

## Overview
This project is a **Customer Churn Prediction** analysis for a telecommunications company using **Python** and machine learning techniques. The goal is to predict whether a customer will churn (i.e., leave the service) based on their demographic, service usage, and contract information. 

We employ **Random Forest Classifier** for predictive modeling, alongside **Exploratory Data Analysis (EDA)** to understand the data and key factors influencing churn.

---

## Key Features
- **Dataset**: [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) dataset.
- **Technologies**: Python, Pandas, Numpy, Scikit-learn, Seaborn, Matplotlib.
- **Analysis Goals**:
  - Identify significant features contributing to customer churn.
  - Visualize key insights using effective plots.
  - Build a predictive machine learning model for churn classification.

---

## Data Preprocessing
1. **Handling Missing Values**:
   - Converted `TotalCharges` to numeric and dropped rows with missing values.
2. **Encoding Target Variable**:
   - `Churn` was converted to binary (1 for "Yes", 0 for "No").
3. **Encoding Categorical Variables**:
   - Dummy variables were created for all categorical features using `pd.get_dummies`.
4. **Scaling**:
   - Used `MinMaxScaler` to scale all features between 0 and 1.

---

## Exploratory Data Analysis (EDA)
### Insights from EDA:
1. **Tenure vs. Churn**:
   - Customers with shorter tenure are more likely to churn. This indicates dissatisfaction or unmet expectations among newer customers.
   - ![Tenure vs Churn](images/Tenure_vs_Churn.png)

2. **Churn by Contract Type**:
   - Customers with month-to-month contracts are most likely to churn due to the lack of long-term commitment.
   - ![Churn by Contract Type](images/Churn_by_Contract_Type.png)

3. **Monthly Charges vs. Churn**:
   - Higher monthly charges are associated with higher churn rates, highlighting pricing dissatisfaction.
   - ![Monthly Charges vs Churn](images/MonthlyCharges_vs_Churn.png)

4. **Total Charges vs. Churn**:
   - Customers with higher total charges are more likely to churn.
   - ![Total Charges vs Churn](images/TotalCharges_vs_Churn.png)

---

## Predictive Modeling
### Steps:
1. **Model**: Built a **Random Forest Classifier** with the following parameters:
   - `n_estimators=1000`
   - `max_leaf_nodes=30`
   - `oob_score=True`
2. **Performance Evaluation**:
   - Achieved high **accuracy** and **recall** scores.
   - Example Metrics:
     ```
     Classification Report:
              precision    recall  f1-score   support

           0       0.84      0.92      0.88      1052
           1       0.66      0.49      0.56       355

     Accuracy: 0.81
     ```

---

## Feature Importance
### Key Features Contributing to Churn:
1. **Contract Type**: Month-to-month contracts have the highest impact on churn.
2. **Tenure**: Short tenure strongly correlates with churn.
3. **Total Charges**: Higher accumulated charges increase churn likelihood.
4. **Online Security & Tech Support**: Lack of these services contributes to churn.

---

## Next Steps
- **Improving Model**:
  - Test additional algorithms like Gradient Boosting or XGBoost.
  - Experiment with hyperparameter tuning.

