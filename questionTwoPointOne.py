# predicting the gamma value
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split


dataSet = pd.read_csv('./data/SmartBuild.csv')

X = dataSet[["Durchmesser", "Hoehe", "Gewicht"]]
y = dataSet.Gammawert
y.to_csv('Gammawert.csv')


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
model = XGBRegressor()
model.fit(Xtrain,Ytrain)
pred = model.predict(Xtest)
pred

plt.scatter(Ytest,pred)
plt.axline([0, 0], [1, 1], dashes=[6, 2], c='red', alpha=0.5)
plt.xlabel("actual")
plt.ylabel("predicted")

mae = metrics.mean_absolute_error(Ytest, pred)
mse = metrics.mean_squared_error(Ytest, pred)
print('MAE is {}'.format(mae))
print('MSE is {}'.format(mse))

pred_resahped=pred.reshape(-1,1)
err = np.subtract(pred_resahped,Ytest.values)

print(stats.describe(err,axis=0))

plt.show()