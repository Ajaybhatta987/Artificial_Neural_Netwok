# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 09:41:23 2021

@author: Ajay
"""

import tensorflow as tf
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Checking columns list and missing values
dataset.isnull().sum()

#Exploratory data analysis
labels = 'Exited', 'Stayed'
sizes = [dataset.Exited[dataset['Exited']==1].count(), dataset.Exited[dataset['Exited']==0].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True)
ax1.axis('equal')
plt.title("Proportion of customer exited and stayed", size = 20)
plt.show()

#Checking the 'Status' relation with categorical variables
fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
sns.countplot(x='Geography', hue = 'Exited',data = dataset, ax=axarr[0][0])
sns.countplot(x='Gender', hue = 'Exited',data = dataset, ax=axarr[0][1])
sns.countplot(x='HasCrCard', hue = 'Exited',data = dataset, ax=axarr[1][0])
sns.countplot(x='IsActiveMember', hue = 'Exited',data = dataset, ax=axarr[1][1])

# Relations based on the continuous data attributes
fig, axarr = plt.subplots(3, 2, figsize=(20, 12))
sns.boxplot(y='CreditScore',x = 'Exited', hue = 'Exited',data = dataset, ax=axarr[0][0])
sns.boxplot(y='Age',x = 'Exited', hue = 'Exited',data = dataset , ax=axarr[0][1])
sns.boxplot(y='Tenure',x = 'Exited', hue = 'Exited',data = dataset, ax=axarr[1][0]) 
sns.boxplot(y='Balance',x = 'Exited', hue = 'Exited',data = dataset, ax=axarr[1][1])
sns.boxplot(y='NumOfProducts',x = 'Exited', hue = 'Exited',data = dataset, ax=axarr[2][0])
sns.boxplot(y='EstimatedSalary',x = 'Exited', hue = 'Exited',data = dataset, ax=axarr[2][1])

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing  import LabelEncoder 
le = LabelEncoder()
X[:, 2]  = le.fit_transform(X[:, 2])    
print(X)

# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X = X[: , 1:]
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Building the ANN
#from tf.keras.layers import Dropout
# Initializing the ANN

ann = tf.keras.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation = 'relu'))
#ann.add(Dropout(p = 0.1))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#ann.add(Dropout(p = 0.1))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN

# Compiling the ANN(https://www.tensorflow.org/guide/keras/train_and_evaluate#the_compile_method_specifying_a_loss_metrics_and_an_optimizer)
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ft = ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Visualizing the loss for each epoch
plt.plot(ft.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)


#making confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)


# Visualize the confusion matrix
sns.heatmap(cm, annot=True , fmt = ".4g")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


#ann_viz(ann ,view = True, title = "Artificial neural network")
#Evaluating the ann
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    ann = tf.keras.Sequential()
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return ann
ann = KerasClassifier(build_fn = build_classifier, batch_size=32, epochs=100)
accuracies = cross_val_score(estimator = ann, X = X_train,y = y_train, cv=10, n_jobs = 1 ,scoring = 'accuracy')


                                      #Completed














#tuning the ann
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    ann = tf.keras.Sequential()
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    ann.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return ann
ann = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size' :[25, 32],
              'nb_epoch' : [100, 500],
              'optimizer' : ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = ann, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_







