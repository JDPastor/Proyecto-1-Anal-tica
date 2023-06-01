# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:24:25 2023

@author: juand
"""

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,cross_validate

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm  # Import tqdm
import pickle

df = pd.read_csv('DatosP3.csv')

print(pd.isnull(df).any())
print(df.shape)

descripcion=df.describe(include='all')

df.dropna(inplace=True)


print(df.dtypes)
PERIODO= df["PERIODO"].unique()
COLE_BILINGUE= df["COLE_BILINGUE"].unique()
COLE_CALENDARIO=df["COLE_CALENDARIO"].unique()
COLE_DEPTO_UBICACION=df["COLE_DEPTO_UBICACION"].unique()
COLE_GENERO=df["COLE_GENERO"].unique()
COLE_NATURALEZA=df["COLE_NATURALEZA"].unique()
ESTU_GENERO=df["ESTU_GENERO"].unique()
COLE_AREA_UBICACION=df["COLE_AREA_UBICACION"].unique()
FAMI_ESTRATOVIVIENDA=df["FAMI_ESTRATOVIVIENDA"].unique()



df['PERIODO'] = df['PERIODO'].astype(object)
categorical_columns = df.select_dtypes(include=['object']).columns

# Perform one-hot encoding for each categorical column
for column in categorical_columns:
    one_hot_encoded = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, one_hot_encoded], axis=1)
    df = df.drop([column], axis=1)
    
    
X = df.drop('PUNT_GLOBAL', axis=1)  # Remove the target variable column from the predictors
y = df['PUNT_GLOBAL']  # Set the target variable column as the target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    

import xgboost as xgb
import pickle
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score

# Set the hyperparameters for the XGBoost model
params = {
    'n_estimators': 100,     # Number of boosting rounds
    'max_depth': 3,          # Maximum tree depth
    'learning_rate': 0.1,    # Learning rate
    'objective': 'reg:squarederror'   # Regression objective
}

# Initialize the XGBoost model
xgb_model = xgb.XGBRegressor(**params)

# Train the XGBoost model
xgb_model.fit(X_train, y_train)

# Serialize the trained model using pickle
with open('xgb_model.pkl', 'wb') as file:
    pickle.dump(xgb_model, file)

# Make predictions on the test dataset
y_pred = xgb_model.predict(X_test)

# Perform evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2) Score:", r2)

xgb.plot_importance(xgb_model, max_num_features=10)
plt.show()

nombresVariables= X_test.columns
x_prediccion=pd.DataFrame(columns=nombresVariables)
x_prediccion.loc[0] = 0




# Print the resulting DataFrame
print(nombresVariables)

