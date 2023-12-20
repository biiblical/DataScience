import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df = pd.read_csv("C:/Users/tring/OneDrive/Desktop/manufacturing_Task_01.csv", sep=",")

# Display basic information about the dataset
print(df.info())
# Display statistical summary of the dataset
print(df.describe())
# Display the shape of the dataset
print(df.shape)
# Check for missing values
print(df.isnull().sum()) # there are none


# One-hot encode 'error_type'
df_encoded = pd.get_dummies(df, columns=['error_type'], drop_first=False)

# Display the updated DataFrame
print(df_encoded.head())
# Display information about the encoded DataFrame
print(df_encoded.info())
# Display statistical summary of the encoded DataFrame
print(df_encoded.describe())
# Display the shape of the encoded DataFrame
print(df_encoded.shape)
print(df_encoded.columns)

# Exclude attributes
X = df_encoded.drop(['id','nicesness','weight_in_g','weight_in_kg', 'FluxCompensation', 'ionizationclass', 'error','error_type_None','error_type_critical','error_type_minor','error_type_severe','multideminsionality','distortion','Quality','reflectionScore'], axis=1)
y = df_encoded['weight_in_kg']

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the data and transform the entire DataFrame
X_scaled = scaler.fit_transform(X)

# Convert the scaled array back to a DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Regressor
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

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

# Make predictions on the test set
rf_y_pred = rf_model.predict(X_test)

# Create a scatter plot of actual vs. predicted values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=rf_y_pred)
plt.title('Scatter Plot of Actual vs. Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# Add a diagonal line for perfect prediction
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)

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

