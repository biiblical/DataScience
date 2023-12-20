import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, tree , linear_model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

df = pd.read_csv('./data/SmartBuild.csv')
columns = ['Durchmesser','Hoehe','Gewicht','Qualitaet']
df = df.loc[:, columns]
features = ['Durchmesser','Hoehe','Gewicht']
target = ['Qualitaet']

X = df.loc[:, features]
y = df.loc[:, target]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size = .80)

model = XGBRegressor()
model.fit(X_train, y_train)
predictions3 = reg.predict(X_test)
score = model.score(X_test, y_test)
print(score)
plt.scatter(y_test, predictions3)
plt.axline([0, 0], [1, 1], dashes=[6, 2], c='red', alpha=0.5)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
mae = metrics.mean_absolute_error(y_test, predictions3)
mse = metrics.mean_squared_error(y_test, predictions3)
print('MAE is {}'.format(mae))
print('MSE is {}'.format(mse))