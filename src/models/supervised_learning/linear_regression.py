import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from time import time


def linear_regression(x, y, test_size=0.3, log_time=True):
    val = []
    header = ['R2 Score', 'Training Time', 'Predict Time']
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=24)

    linReg = LinearRegression()
    start = time()
    linReg.fit(X_train, y_train)
    end_train = time()

    y_pred = linReg.predict(X_test)
    end_pred = time()

    val.append([r2_score(y_test, y_pred), end_train -
                start, end_pred - end_train])
    return tabulate(val, headers=header)
    # print('R2 Score: ', r2_score(y_test, y_pred))
    # if log_time:
    #     print(f'Training Time: {end_train - start}s')
    #     print(f'Predict Time: {end_pred - end_train}s')


# student = pd.read_csv('data/student/student-mat.csv')
# y = student['G3']

# print('\n-----Label Encoded-----')
# df = preprocessing(data=student, y=y)
# linear_regression(df.loc[:, df.columns.difference(['G3'])], df['G3'])


# print('\n----One Hot Encoded-----')
# df = preprocessing(data=student, y=y, perform_ohe=True)
# linear_regression(df.loc[:, df.columns.difference(['G3'])], df['G3'])


# print('\n----Standard Scaled-----')
# df = preprocessing(data=student, y=y, perform_ohe=True, perform_scale=True)
# linear_regression(df.loc[:, df.columns.difference(['G3'])], df['G3'])


# print('\n---Feature Selection----')
# df = preprocessing(data=student, y=y, perform_scale=True)
# df = feature_selection(df=df, target=y)
# linear_regression(df.loc[:, df.columns.difference(['G3'])], df['G3'])


# print('\n---------PCA------------')
# df = preprocessing(data=student, y=y, perform_scale=True)
# df = pca(df.loc[:, df.columns.difference(['G3'])], df['G3'], 0.9)
# linear_regression(df.loc[:, df.columns.difference(['G3'])], df['G3'])

# print(tabulate(val, headers=header))
