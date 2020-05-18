import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from time import time


def logistic_regression(x, y, test_size=0.3, max_iter=500, log_time=True):
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=24)

    logReg = LogisticRegression(
        solver='lbfgs', multi_class='multinomial', random_state=24, max_iter=max_iter)
    start = time()
    logReg.fit(X_train, y_train)
    end_train = time()

    y_pred = logReg.predict(X_test)
    end_pred = time()

    print('Accuracy Score: ', accuracy_score(y_test, y_pred))
    print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
    if log_time:
        print(f'Training Time: {end_train - start}s')
        print(f'Predict Time: {end_pred - end_train}s')


df = pd.read_csv('data/student/student-mat-fully-processed.csv')

df['G3'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

# All variables used
print('\nAll variables used')
logistic_regression(df.loc[:, df.columns.difference(['G3'])], df['G3'])

# All variables used except G2
print('\nAll variables used except G2')
logistic_regression(df.loc[:, df.columns.difference(['G2', 'G3'])], df['G3'])

# All variables used except G1 G2
print('\nAll variables used except G1 G2')
logistic_regression(
    df.loc[:, df.columns.difference(['G1', 'G2', 'G3'])], df['G3'])

# Output:
# All variables used
# Accuracy Score:  0.9495798319327731
# Confusion Matrix:
#  [[32  2]
#  [ 4 81]]
# Training Time: 0.06660985946655273s
# Predict Time: 0.0009162425994873047s

# All variables used except G2
# Accuracy Score:  0.8151260504201681
# Confusion Matrix:
#  [[30  4]
#  [18 67]]
# Training Time: 0.05706286430358887s
# Predict Time: 0.0007030963897705078s

# All variables used except G1 G2
# Accuracy Score:  0.680672268907563
# Confusion Matrix:
#  [[17 17]
#  [21 64]]
# Training Time: 0.02388310432434082s
# Predict Time: 0.0008029937744140625s
