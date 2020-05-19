import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from time import time


def decision_tree(x, y, test_size=0.3, log_time=True):
    val = []
    header = ['Accuracy Score', 'Confusion Matrix',
              'Training Time', 'Predict Time']
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=24)

    dect = DecisionTreeClassifier(
        criterion='gini', max_depth=None, min_samples_leaf=1, min_samples_split=2, random_state=24)
    start = time()
    dect.fit(X_train, y_train)
    end_train = time()

    y_pred = dect.predict(X_test)
    end_pred = time()

    val.append([accuracy_score(y_test, y_pred), confusion_matrix(
        y_test, y_pred), end_train - start, end_pred - end_train])
    return tabulate(val, headers=header)
    # print('Accuracy Score: ', accuracy_score(y_test, y_pred))
    # print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
    # if log_time:
    #     print(f'Training Time: {end_train - start}s')
    #     print(f'Predict Time: {end_pred - end_train}s')


# student = pd.read_csv('data/student/student-mat.csv')
# student['G3'] = student['G3'].apply(lambda x: 1 if x >= 10 else 0)
# y = student['G3']

# print('\n-----Label Encoded-----')
# df = preprocessing(data=student, y=y)
# decision_tree(df.loc[:, df.columns.difference(['G3'])], df['G3'])


# print('\n----One Hot Encoded-----')
# df = preprocessing(data=student, y=y, perform_ohe=True)
# decision_tree(df.loc[:, df.columns.difference(['G3'])], df['G3'])


# print('\n----Standard Scaled-----')
# df = preprocessing(data=student, y=y, perform_ohe=True, perform_scale=True)
# decision_tree(df.loc[:, df.columns.difference(['G3'])], df['G3'])


# print('\n---Feature Selection----')
# df = preprocessing(data=student, y=y, perform_scale=True)
# df = feature_selection(df=df, target=y)
# decision_tree(df.loc[:, df.columns.difference(['G3'])], df['G3'])


# print('\n---------PCA------------')
# df = preprocessing(data=student, y=y, perform_scale=True)
# df = pca(df.loc[:, df.columns.difference(['G3'])], df['G3'], 0.9)
# decision_tree(df.loc[:, df.columns.difference(['G3'])], df['G3'])

# print(tabulate(val, headers=header))

# print('\n-----Decision Tree-----')

# mushroom = pd.read_csv('data/mushroom/mushrooms.csv')
# y = mushroom['class']

# # print('\n-----Label Encoded-----')
# df = preprocessing(data=mushroom, y=y)
# decision_tree(df.loc[:, df.columns.difference(['class'])], df['class'])


# # print('\n----One Hot Encoded-----')
# df = preprocessing(data=mushroom, y=y, perform_ohe=True)
# decision_tree(df.loc[:, df.columns.difference(['class'])], df['class'])


# # print('\n----Standard Scaled-----')
# df = preprocessing(data=mushroom, y=y, perform_ohe=True, perform_scale=True)
# decision_tree(df.loc[:, df.columns.difference(['class'])], df['class'])


# # print('\n---Feature Selection----')
# df = preprocessing(data=mushroom, y=y, perform_scale=True)
# df = feature_selection(df=df, target=y)
# decision_tree(df.loc[:, df.columns.difference(['class'])], df['class'])


# # print('\n---------PCA------------')
# df = preprocessing(data=mushroom, y=y, perform_scale=True)
# df = pca(df.loc[:, df.columns.difference(['class'])], df['class'], 0.9)
# decision_tree(df.loc[:, df.columns.difference(['class'])], df['class'])

# print(tabulate(val, headers=header))
