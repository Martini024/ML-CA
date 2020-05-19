import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras import layers
from sklearn.metrics import accuracy_score, confusion_matrix
from time import time
from preprocessing.preprocessing import preprocessing
from preprocessing.feature_selection import feature_selection
from preprocessing.pca import pca


val = []
header = ['Accuracy Score', 'Confusion Matrix',
          'Training Time', 'Predict Time']


def neural_network(x, y, test_size=0.3, log_time=True):

    X_train, X_test, y_train, y_test = train_test_split(
        x.values, y.values, test_size=0.3, random_state=42)
    y_train = np.array([[0] if x == 0 else [1] for x in y_train])
    y_test = np.array([[0] if x == 0 else [1] for x in y_test])

    model = Sequential(
        [
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ]
    )
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=["accuracy"])
    start = time()
    model.fit(X_train, y_train, batch_size=64, epochs=100, verbose=1)
    end_train = time()

    # perform auto-evaluation
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print('Loss = ', loss, ', Accuracy = ', accuracy)

    # perform prediction (let's eye-ball the results)
    y_pred = model.predict(X_test)
    end_pred = time()
    # for i in np.arange(len(predictions)):
    #     print('Actual: ', y_test_ohe[i], ', Predicted: ', predictions[i])

    val.append([accuracy_score(y_test, y_pred.round()), confusion_matrix(
        y_test, y_pred.round()), end_train - start, end_pred - end_train])
    print('Accuracy Score: ', accuracy_score(y_test, y_pred.round()))
    print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred.round()))
    if log_time:
        print(f'Training Time: {end_train - start}s')
        print(f'Predict Time: {end_pred - end_train}s')


student = pd.read_csv('data/student/student-mat.csv')
student['G3'] = student['G3'].apply(lambda x: 1 if x >= 10 else 0)
y = student['G3']

print('\n-----Label Encoded-----')
df = preprocessing(data=student, y=y)
neural_network(df.loc[:, df.columns.difference(['G3'])], df['G3'])


print('\n----One Hot Encoded-----')
df = preprocessing(data=student, y=y, perform_ohe=True)
neural_network(df.loc[:, df.columns.difference(['G3'])], df['G3'])


print('\n----Standard Scaled-----')
df = preprocessing(data=student, y=y, perform_ohe=True, perform_scale=True)
neural_network(df.loc[:, df.columns.difference(['G3'])], df['G3'])


print('\n---Feature Selection----')
df = preprocessing(data=student, y=y, perform_scale=True)
df = feature_selection(df=df, target=y)
neural_network(df.loc[:, df.columns.difference(['G3'])], df['G3'])


print('\n---------PCA------------')
df = preprocessing(data=student, y=y, perform_scale=True)
df = pca(df.loc[:, df.columns.difference(['G3'])], df['G3'], 0.9)
neural_network(df.loc[:, df.columns.difference(['G3'])], df['G3'])

print(tabulate(val, headers=header))
