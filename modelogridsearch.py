# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 08:53:46 2023

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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Set the hyperparameters for the XGBoost model
params = {
    'n_estimators': [100, 200, 300],     # Number of boosting rounds
    'max_depth': [3, 4, 5],               # Maximum tree depth
    'learning_rate': [0.1, 0.01, 0.001],  # Learning rate
    'objective': ['reg:squarederror']      # Regression objective
}

# Initialize the XGBoost model
xgb_model = xgb.XGBRegressor()

# Perform grid search with custom progress tracking
param_combinations = len(params['n_estimators']) * len(params['max_depth']) * len(params['learning_rate'])
current_combination = 0
best_mse = float('inf')
best_model = None

for n_estimators in params['n_estimators']:
    for max_depth in params['max_depth']:
        for learning_rate in params['learning_rate']:
            current_combination += 1
            print(f"Training model {current_combination}/{param_combinations} - n_estimators: {n_estimators}, max_depth: {max_depth}, learning_rate: {learning_rate}")

            # Set the hyperparameters for the XGBoost model
            xgb_model.set_params(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)

            # Train the XGBoost model
            xgb_model.fit(X_train, y_train)

            # Make predictions on the test dataset
            y_pred = xgb_model.predict(X_test)

            # Perform evaluation
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Print the evaluation metrics
            print("Mean Squared Error (MSE):", mse)
            print("R-squared (R2) Score:", r2)

            # Check if the current model has the lowest MSE so far
            if mse < best_mse:
                best_mse = mse
                best_model = xgb_model

# Serialize the best model using pickle
with open('xgb_model_grid.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# Make predictions on the test dataset using the best model
y_pred = best_model.predict(X_test)

# Perform evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2) Score:", r2)

with open("xgb_model_grid.pkl", 'rb') as file:
    loaded_model = pickle.load(file)

xgb.plot_importance(loaded_model)
plt.show()
