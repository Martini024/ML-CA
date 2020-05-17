import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras import layers
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('data/mushrooms.csv')

df = df.apply(lambda x: pd.factorize(x)[0])

X_train, X_test, y_train, y_test = train_test_split(
    df.loc[:, df.columns != 'class'].values, df['class'].values, test_size=0.3, random_state=24)
y_train = np.array([[0] if x == 0 else [1] for x in y_train])
y_test = np.array([[0] if x == 0 else [1] for x in y_test])

model = Sequential(
    [
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ]
)
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=["binary_accuracy"])
model.fit(X_train, y_train, batch_size=64, epochs=5, verbose=1)

# perform auto-evaluation
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print('Loss = ', loss, ', Accuracy = ', accuracy)

# perform prediction (let's eye-ball the results)
y_pred = model.predict(X_test)
# for i in np.arange(len(predictions)):
#     print('Actual: ', y_test_ohe[i], ', Predicted: ', predictions[i])

print(accuracy_score(y_test, y_pred.round()))
print(confusion_matrix(y_test, y_pred.round()))
