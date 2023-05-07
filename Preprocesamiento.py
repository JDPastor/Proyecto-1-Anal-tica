# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 12:42:03 2023
@author: juand
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('processed.cleveland.data', header=None,names=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num"])


#Análisis exploratorio de los datos
df.replace('?', np.nan, inplace=True)
print(df)
print(pd.isnull(df).any())
print(df.shape)
df.dropna(inplace=True)
print(df.describe())

# Plot a histogram for each feature in the dataframe
df.hist(bins=50, figsize=(30,30))
plt.show()

# Plot a heatmap to visualize the correlation between features
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=False)
plt.show()

#Binarización de variables
df['cp'] = df['cp'].apply(lambda x: 0 if x == 4 else 1)
df['restecg'] = df['restecg'].apply(lambda x: 1 if x >= 1 else 0)
df['trestbps'] = df['trestbps'].apply(lambda x: 1 if x >= 120 else 0)
df['chol'] = df['chol'].apply(lambda x: 1 if x >= 240 else 0)
for i in df.index:
    if df.loc[i, 'thalach'] >= (220 - df.loc[i, 'age']):
        df.loc[i, 'thalach'] = 1
    else:
        df.loc[i, 'thalach'] = 0
df['age'] = df['age'].apply(lambda x: 1 if x >= 45 else 0)
df.drop('oldpeak',axis=1, inplace=True)
df['slope'] = df['slope'].apply(lambda x: 1 if x == 3 else 0)
df['thal'] = df['thal'].apply(lambda x: 0 if x == '3.0' else 1)
df['ca'] = df['ca'].apply(lambda x: 0 if x == '0.0' else 1)
df['num']=df['num'].apply(lambda x:0 if x==0 else 1)

df = df.drop('ca',axis = 1)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv('datosTrain',index = False)
test_df.to_csv('datosTest',index = False)
