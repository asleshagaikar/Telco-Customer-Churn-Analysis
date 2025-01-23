import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# Load dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(df.head())
print(df.columns.values)
# Checking the data types of all the columns
print(df.dtypes)

#-------------------------------Preprocessing-------------------------------
# Converting Total Charges to a numerical data type and Handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(df.isnull().sum())
#df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df.dropna(inplace = True)

#Remove customer IDs from the data set
df2 = df.iloc[:,1:]
#Convertin the predictor variable in a binary numeric variable
df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)
df2['Churn'].replace(to_replace='No',  value=0, inplace=True)

#Let's convert all the categorical variables into dummy variables 
#Creates a new column for each unique category value, with binary values (0 or 1) indicating the presence of the category.
df_dummies = pd.get_dummies(df2)
print(df_dummies.head())
print(df_dummies.columns)

#------------------------------- EDA (Exploratory Data Analysis)-------------------------------
#1) Churn vs tenure
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title('Tenure vs. Churn')
plt.savefig('Tenure_vs_Churn.png')
# plt.show()

#2)Churn vs. Contract Type
colors = ['#4D3425','#E4512B']
contract_churn = df.groupby(['Contract','Churn']).size().unstack()

ax = (contract_churn.T*100.0 / contract_churn.T.sum()).T.plot(kind='bar',width = 0.3,stacked = True,rot = 0, figsize = (10,6),color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='best',prop={'size':14},title = 'Churn')
ax.set_ylabel('% Customers',size = 14)
ax.set_title('Churn by Contract Type',size = 14)

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
                color = 'white',
               weight = 'bold',
               size = 14)
# Save the figure as a PNG file
plt.savefig('Churn_by_Contract_Type.png', format='png', dpi=300)

#3) MonthlyCharges vs. Churn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('MonthlyCharges vs. Churn')
plt.savefig('MonthlyCharges_vs_Churn.png')
# plt.show()

#4) TotalCharges vs. Churn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='TotalCharges', data=df)
plt.title('TotalCharges vs. Churn')
plt.savefig('TotalCharges_vs_Churn.png')
# plt.show()


#------------------------------- Predictive Modeling -------------------------------

# Define features and target variable
X = df_dummies.drop(columns = ['Churn'])
y = df_dummies['Churn']

# Scaling all the variables to a range of 0 to 1
features = X.columns.values
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = features

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)



# Train a RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=1000 , oob_score = True, n_jobs = -1, random_state =50,max_leaf_nodes = 30)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print (accuracy_score(y_test, y_pred))

# Feature importance analysis
# Assuming importances and features are already defined
importances = rf_model.feature_importances_
features = X.columns

# Create a dataframe with features and their importances
data = pd.DataFrame({'features': features, 'importances': importances})

# Sort the data by importances in descending order
data_sorted = data.sort_values(by='importances', ascending=False)

# Plot the feature importances
plt.figure(figsize=(15, 10))
sns.barplot(x='importances', y='features', data=data_sorted)
plt.title('Feature Importances')
plt.savefig('FeatureImportance.png')
#plt.show()