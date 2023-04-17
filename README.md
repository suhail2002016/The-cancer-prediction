
# The-cancer-prediction

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('/content/sample_data/cancer patient data sets (1).csv')
df.isnull().sum()
df = pd.read_csv('/content/sample_data/cancer patient data sets (1).csv')
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 2})
sns.barplot(x=df['Gender'],y=df['Alcohol use']) plt.xticks(rotation='vertical')
plt.show()
sns.boxplot(x='Gender', y='Age', data=df) plt.show()
sns.displot(df['Level'].dropna())

 <seaborn.axisgrid.FacetGrid at 0x7f8fed2a0220>
df.drop('Patient Id', axis=1, inplace=True) df.head()
import pandas as pd
# assume this is your input data data = pd.DataFrame({
'Age': [30, 40, 20, 25],
4 5 1 5 5 6 7 7 8 7
4 3 3 4 5 5 7 6 7 7
0
1
2
3
4
'OccuPational Hazards': ['Engineer', 'Teacher', 'Doctor', 'Engineer'],
'chronic Lung Disease': ['1', '0', '0', '1'] })
# define the categorical features to be one-hot encoded
categorical_features = ['Age', 'OccuPational Hazards', 'chronic Lung Disease']
# use pandas get_dummies method to one-hot encode the categorical features encoded_data = pd.get_dummies(data, columns=categorical_features)
# print the encoded data

print(encoded_data.head())
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Define the data and the features
data = pd.read_csv('/content/sample_data/cancer patient data sets (1).csv') numerical_features = ['Age', 'Air Pollution', 'Alcohol use', 'Dust Allergy', 'Genetic R
'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker', 'Chest Pa 'Coughing of Blood', 'Fatigue', 'Weight Loss', 'Shortness of Breat 'Wheezing', 'Swallowing Difficulty', 'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring']
# Separate the input features and the target variable X = data[numerical_features]
y = data['Level']
# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 from sklearn.model_selection import train_test_split
# Split data into features and target variable X = data[categorical_features]
y = data['Level']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42
# Define the categorical features and one-hot encode them
categorical_features = ['Age', 'OccuPational Hazards', 'chronic Lung Disease'] X_train_encoded = pd.get_dummies(X_train, columns=categorical_features) X_test_encoded = pd.get_dummies(X_test, columns=categorical_features)

from sklearn.model_selection import cross_val_score from sklearn.linear_model import LogisticRegression from sklearn.datasets import load_iris
# Load the iris dataset iris = load_iris()
# Create X and y variables X = iris.data
y = iris.target
# Create a logistic regression model model = LogisticRegression()
# Perform cross-validation with 5 folds cv_scores = cross_val_score(model, X, y, cv=5)
# Print the cross-validation scores for each fold print("Cross-validation scores:", cv_scores)
# Print the average cross-validation score
print("Average cross-validation score:", cv_scores.mean())
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
# Define the pre-processing steps
numerical_features = ['Age', 'Air Pollution', 'Alcohol use', 'Dust Allergy', 'Genetic Ri
'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker', 'Chest Pa 'Coughing of Blood', 'Fatigue', 'Weight Loss', 'Shortness of Breat 'Wheezing', 'Swallowing Difficulty', 'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring']
categorical_features = ['Gender', 'Occupation', 'Mode of transport']
# Create a pre-processing pipeline for numerical and categorical features numerical_transformer = StandardScaler()
categorical_transformer = LabelEncoder()
preprocessor = ColumnTransformer( transformers=[
('num', numerical_transformer, numerical_features), ('cat', categorical_transformer, categorical_features)])
# Create a pipeline with pre-processing and model
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
 
('classifier', RandomForestClassifier(n_estimators=100, ra

