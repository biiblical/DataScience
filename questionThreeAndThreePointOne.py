from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
from yellowbrick.classifier import ClassPredictionError


def predictLscore():
    data = pd.read_csv('SmartBuild.csv')
    model = DecisionTreeClassifier(max_depth = 4)
    # Select only input data
    X = data[["Durchmesser", "Hoehe", "Gewicht"]]
    # Only select out value to predict
    Y = data['LScore']
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=True, stratify=Y)
    # Train on training data only.
    model.fit(X_train, y_train)

    # Model ready for evaluation.

    import matplotlib.pyplot as plt
    # Plot tree chart.
    plt.figure(figsize=(30, 10))
    tree.plot_tree(model, feature_names=X.columns, class_names=model.classes_, fontsize=13, filled=True)
    plt.show()

    # Create a 3 way confusion matrix. ON TEST DATA ONLY.
    confusionMatrix = metrics.confusion_matrix(y_test, model.predict(X_test))
    # Print matrix
    print(confusionMatrix)
    # Create 3 way plot confusion matrix
    plot_confusion_matrix(model, X_test, y_test)
    plt.show()
    # Print classification report for evaluation
    print(metrics.classification_report(y_test, model.predict(X_test)))
    # REFERENCES
    # https://towardsdatascience.com/how-to-implement-and-evaluate-decision-tree-classifiers-from-scikit-learn-36ef7f037a78
    # https://medium.com/apprentice-journal/evaluating-multi-class-classifiers-12b2946e755b
    # (Goonewardana, 2019)
    # (Wilkinson, 2022)

def XKlassePrediction():
    data = pd.read_csv('SmartBuild.csv')
    model = LogisticRegression()
    # Select only input variables.
    X = data[["Durchmesser", "Hoehe", "Gewicht"]]
    # Select only output variables.
    Y = data['XKlasse']
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=42)

    # Fit model Logistic Regression
    model.fit(X_train, y_train)

    # 3 way confusion matrix Logistic Regression.
    plot_confusion_matrix(model, X_test, y_test)
    plt.show()

    print("-------LOGISTIC REGRESSION START---------")
    # Print confusion matrix Logistic Regression.
    confusionMatrix = metrics.confusion_matrix(y_test, model.predict(X_test))
    print(confusionMatrix)
    # Print classification report Logistic Regression.
    print(metrics.classification_report(y_test, model.predict(X_test)))

    print("-------LOGISTIC REGRESSION END---------")
    # Create yellow brick random forest classifier
    # Classes of values
    classes = ["I", "II", "III", "IV"]
    visualizer = ClassPredictionError(RandomForestClassifier(random_state=42, n_estimators=10), classes=classes)
    # Fit
    visualizer.fit(X_train, y_train)
    # Evaluate
    visualizer.score(X_test, y_test)
    # Show
    visualizer.show()

if __name__ == '__main__':
    predictLscore()
    XKlassePrediction()

    # Goonewardana, H. (2019). Evaluating Multi-Class Classifiers. [online] Apprentice Journal. Available at: https://medium.com/apprentice-journal/evaluating-multi-class-classifiers-12b2946e755b [Accessed 8 Jun. 2022].
    # ‌
    # Wilkinson, P. (2022). How to Implement and Evaluate Decision Tree classifiers from scikit-learn. [online] Medium. Available at: https://towardsdatascience.com/how-to-implement-and-evaluate-decision-tree-classifiers-from-scikit-learn-36ef7f037a78 [Accessed 8 Jun. 2022].
    # ‌
