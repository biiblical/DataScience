import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


print('Hello world')
# Read the dataset into a dataframe
dataSet = pd.read_csv('./data/SmartBuild.csv')

# Select Durchmesser, Hoehe and Gewicht as the X data.
X = dataSet[["Durchmesser", "Hoehe", "Gewicht"]]
# Set the Y data to Y data
y = dataSet.Fehler == 'nein'

# Save the X and Y data to csv files, so I can check the data quality
X.to_csv('FilteredDataInput.csv')
y.to_csv('YdataInput.csv')

# Split the train and test data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=2, shuffle=True)

model = XGBClassifier()
model.fit(Xtrain, Ytrain)
pred = model.predict(Xtest)

# Print the confusion matrix data
print(confusion_matrix(Ytest, pred))

tn, fp, fn, tp = confusion_matrix(Ytest, pred).ravel()
pred_proba = model.predict_proba(Xtest)
pred_proba
fpr, tpr, thresholds = metrics.roc_curve(Ytest, pred_proba[:, 0])

plt.plot(tpr, fpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.axline([0, 0], [1, 1], dashes=[6, 2], c='red', alpha=0.5)
plt.show()