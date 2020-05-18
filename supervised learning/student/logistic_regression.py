import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('data/student/student-mat-processed.csv')

df['G3'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

# All variables used
print('\nAll variables used')
X_train, X_test, y_train, y_test = train_test_split(
    df.loc[:, df.columns.difference(['G3'])], df['G3'], test_size=0.3, random_state=24)

logReg = LogisticRegression(
    solver='lbfgs', multi_class='multinomial', random_state=24, max_iter=500)
logReg.fit(X_train, y_train)

y_pred = logReg.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# All variables used except G2
print('\nAll variables used except G2')
X_train, X_test, y_train, y_test = train_test_split(
    df.loc[:, df.columns.difference(['G2', 'G3'])], df['G3'], test_size=0.3, random_state=24)

logReg = LogisticRegression(
    solver='lbfgs', multi_class='multinomial', random_state=24, max_iter=500)
logReg.fit(X_train, y_train)

y_pred = logReg.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# All variables used except G1 G2
print('\nAll variables used except G1 G2')
X_train, X_test, y_train, y_test = train_test_split(
    df.loc[:, df.columns.difference(['G1', 'G2', 'G3'])], df['G3'], test_size=0.3, random_state=24)

logReg = LogisticRegression(
    solver='lbfgs', multi_class='multinomial', random_state=24, max_iter=500)
logReg.fit(X_train, y_train)

y_pred = logReg.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Output:
# All variables used
# 0.9495798319327731
# [[32  2]
#  [ 4 81]]

# All variables used except G2
# 0.8151260504201681
# [[30  4]
#  [18 67]]

# All variables used except G1 G2
# 0.680672268907563
# [[17 17]
#  [21 64]]