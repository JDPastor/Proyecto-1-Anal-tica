# -*- coding: utf-8 -*-
"""
Created on Wed May 31 21:29:38 2023

@author: juanf
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
import tensorflow as tf

df = pd.read_csv('DatosP3.csv')

print(pd.isnull(df).any())
print(df.shape)

descripcion=df.describe(include='all')

df.dropna(inplace=True)


print(df.dtypes)

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
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1, random_state=42)

# Define the number of features in your data
num_features = 57

# Define the architecture of the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(num_features,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)  # No activation function for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

# Train the model
model.fit(xtrain, ytrain, epochs=10, batch_size=16, validation_data=(xtest, ytest))

with open('modelnn.pkl', 'wb') as file:
    pickle.dump(model, file)
    


