from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns


# Load your dataset
df = pd.read_csv("C:/Users/tring/OneDrive/Desktop/manufacturing_Task_01.csv", sep=",")

# Assuming 'quality' is your target variable
y_quality = df['Quality']

# Features (X) - Exclude irrelevant or redundant columns
X_quality = df.drop(['id','Quality', 'FluxCompensation', 'ionizationclass', 'error', 'error_type', 'multideminsionality', 'nicesness', 'distortion', 'reflectionScore', 'weight_in_g', 'weight_in_kg'], axis=1)

# Check for missing values in 'quality'
print(df['Quality'].isnull().sum())

# Remove rows with missing or non-numeric values in 'quality'
df_quality = df.dropna(subset=['Quality'])

# Split the data into training and testing sets
X_train_quality, X_test_quality, y_train_quality, y_test_quality = train_test_split(X_quality, y_quality, test_size=0.2, random_state=42)


# Create a Random Forest Regressor model
rf_model_quality = RandomForestRegressor(random_state=42)

# Fit the model to the training data
rf_model_quality.fit(X_train_quality, y_train_quality)

# Make predictions on the testing set
y_pred_quality = rf_model_quality.predict(X_test_quality)

# Calculate Mean Squared Error
mse_quality = mean_squared_error(y_test_quality, y_pred_quality)
print(f'Mean Squared Error (quality): {mse_quality}')

# Calculate R-squared (coefficient of determination)
r2_quality = r2_score(y_test_quality, y_pred_quality)
print(f'R-squared (quality): {r2_quality}')


# Scatter plot for actual vs predicted 'Quality'
plt.scatter(y_test_quality, y_pred_quality, alpha=0.5)
plt.title('Actual vs Predicted Quality')
plt.xlabel('Actual Quality')
plt.ylabel('Predicted Quality')
plt.plot([min(y_test_quality), max(y_test_quality)], [min(y_test_quality), max(y_test_quality)], linestyle='--', color='red', linewidth=2)
plt.show()


importances = rf_model_quality.feature_importances_
feature_names = X_train_quality.columns

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(12, 6))
plt.bar(range(X_train_quality.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train_quality.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Random Forest Regressor - Feature Importances')
plt.show()