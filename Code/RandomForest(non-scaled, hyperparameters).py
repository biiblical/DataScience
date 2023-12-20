import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import pandas as pd
import os

num_cores = os.cpu_count()
print("Number of CPU cores:", num_cores)


# Load your dataset
df = pd.read_csv("C:/Users/Alek/Desktop/UNI/Data Science/manufacturing_Task_01.csv", sep=",")

# One-hot encode 'error_type'
df_encoded = pd.get_dummies(df, columns=['error_type'], drop_first=False)

# Exclude 'FluxCompensation' and 'ionizationclass'
X = df_encoded.drop(['weight_in_kg', 'FluxCompensation', 'ionizationclass', 'error', 'multideminsionality'], axis=1)
y = df_encoded['weight_in_kg']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Create and train the Random Forest Regressor
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Perform GridSearchCV
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Make predictions using the best model
best_rf_model = grid_search.best_estimator_
best_rf_y_pred = best_rf_model.predict(X_test)

# Calculate and print performance metrics for the tuned model
best_rf_mse = mean_squared_error(y_test, best_rf_y_pred)
best_rf_r2 = r2_score(y_test, best_rf_y_pred)

# Make predictions on the test set
rf_y_pred = rf_model.predict(X_test)

# Calculate and print performance metrics
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)

# Print feature importances
feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)

print(f'Random Forest Mean Squared Error: {rf_mse}')
print(f'Random Forest R-squared: {rf_r2}')
print("Feature Importances:")
print(feature_importances)

# Create a scatter plot of actual vs. predicted values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=rf_y_pred)
plt.title('Scatter Plot of Actual vs. Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

# Calculate residuals
residuals = y_test - rf_y_pred

# Create a residual plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')  # Horizontal line at y=0
plt.title('Residual Plot')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.show()


print(f'Tuned Random Forest Mean Squared Error: {best_rf_mse}')
print(f'Tuned Random Forest R-squared: {best_rf_r2}')