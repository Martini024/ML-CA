import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('data/mushroom/mushrooms.csv')

# use pandas factorize to label encoding categorical data (in mushroom dataset means all of them)
# PPT Model Engineering P4
df = df.apply(lambda x: pd.factorize(x)[0])

# split training and test data at ratio 7:3
X_train, X_test, y_train, y_test = train_test_split(
    df.loc[:, df.columns != 'class'], df['class'], test_size=0.3, random_state=24)

# fit model
logReg = LogisticRegression(
    solver='lbfgs', multi_class='multinomial', random_state=24)
logReg.fit(X_train, y_train)

# predict test data
y_pred = logReg.predict(X_test)

# perform evaluation
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
