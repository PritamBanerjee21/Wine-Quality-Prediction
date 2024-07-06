# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('D:/WineQuality deploy/winequality-red.csv')
df.sample(5)

df['quality'] = df['quality'].apply(lambda x : 1 if x>=7 else 0)
df
df['quality'].value_counts()

#Making new feature from two features
df['total acidity'] = df['fixed acidity'] + df['volatile acidity']
df.head()

#Removing unnecessary features
df = df.drop(['fixed acidity','volatile acidity','free sulfur dioxide'],axis=1)
df.head()

#Splitting our data into input and output
X = df.drop(['quality'],axis=1)
y = df['quality']

#importing dependencies
from imblearn.ensemble import BalancedRandomForestClassifier
from scikit-lean.model_selection import train_test_split

#Splitting the dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Training our model
model = BalancedRandomForestClassifier(sampling_strategy="all", replacement=True, random_state=0,bootstrap=False,
                                       criterion = 'gini', max_depth= 7, n_estimators= 50, n_jobs= -1)

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

#Test accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)


#Training accuracy
y_pred_train = model.predict(X_train)
accuracy_score(y_pred_train,y_train)

#Saving our model
from joblib import dump, load
filename = ('WineQualityPredictor.joblib')
dump(model,filename)
