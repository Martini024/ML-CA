import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from time import time
from preprocessing.preprocessing import preprocessing
from preprocessing.feature_selection import feature_selection
from preprocessing.pca import pca

val = []
header = ['Accuracy Score', 'Confusion Matrix',
          'Training Time', 'Predict Time']


def knn(x, y, test_size=0.3, n_neighbors=10, log_time=True):
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=24)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    start = time()
    knn.fit(X_train, y_train)
    end_train = time()

    y_pred = knn.predict(X_test)
    end_pred = time()

    val.append([accuracy_score(y_test, y_pred), confusion_matrix(
        y_test, y_pred), end_train - start, end_pred - end_train])
    print('Accuracy Score: ', accuracy_score(y_test, y_pred))
    print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
    if log_time:
        print(f'Training Time: {end_train - start}s')
        print(f'Predict Time: {end_pred - end_train}s')


student = pd.read_csv('data/student/student-mat.csv')
student['G3'] = student['G3'].apply(lambda x: 1 if x >= 10 else 0)
y = student['G3']

print('\n-----Label Encoded-----')
df = preprocessing(data=student, y=y)
knn(df.loc[:, df.columns.difference(['G3'])], df['G3'])


print('\n----One Hot Encoded-----')
df = preprocessing(data=student, y=y, perform_ohe=True)
knn(df.loc[:, df.columns.difference(['G3'])], df['G3'])


print('\n----Standard Scaled-----')
df = preprocessing(data=student, y=y, perform_ohe=True, perform_scale=True)
knn(df.loc[:, df.columns.difference(['G3'])], df['G3'])


print('\n---Feature Selection----')
df = preprocessing(data=student, y=y, perform_scale=True)
df = feature_selection(df=df, target=y)
knn(df.loc[:, df.columns.difference(['G3'])], df['G3'])


print('\n---------PCA------------')
df = preprocessing(data=student, y=y, perform_scale=True)
df = pca(df.loc[:, df.columns.difference(['G3'])], df['G3'], 0.9)
knn(df.loc[:, df.columns.difference(['G3'])], df['G3'])

print(tabulate(val, headers=header))
