import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv("C:/Users/tring/OneDrive/Desktop/manufacturing_Task_01.csv", sep=",")

# Assuming 'error' is your target variable for binary classification (yes/no)
y_error = df['error']

# Features (X) - Exclude irrelevant or redundant columns
X_error = df.drop(['id', 'Quality', 'nicesness', 'distortion', 'FluxCompensation', 'weight_in_kg', 'weight_in_g', 'error', 'error_type', 'multideminsionality', 'reflectionScore'], axis=1)

# Check for missing values in 'error'
print(df['error'].isnull().sum())

# Remove rows with missing or non-numeric values in 'error'
df_error = df.dropna(subset=['error'])

# One-hot encode categorical features
X_error = pd.get_dummies(X_error)

# Encode binary labels if needed
label_encoder = LabelEncoder()  
y_error_encoded = label_encoder.fit_transform(y_error)

# Split the data into training and testing sets
X_train_error, X_test_error, y_train_error, y_test_error = train_test_split(X_error, y_error_encoded, test_size=0.2, random_state=42)

# Create a Random Forest Classifier model (you can use other classifiers as well)
rf_model_error = RandomForestClassifier(random_state=42)

# Fit the model to the training data
rf_model_error.fit(X_train_error, y_train_error)

# Make predictions on the testing set
y_pred_error = rf_model_error.predict(X_test_error)

# Create a confusion matrix
conf_matrix_error = confusion_matrix(y_test_error, y_pred_error)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_error, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for Error (Yes/No)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot feature importance
feature_importance = rf_model_error.feature_importances_
feature_names = X_error.columns

plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importance, y=feature_names, palette="viridis")
plt.title('Feature Importance for Error Prediction')
plt.xlabel('Feature Importance')
plt.ylabel('Feature Names')
plt.show()

# Print classification report
print("Classification Report:")
print(classification_report(y_test_error, y_pred_error, target_names=['No Error', 'Error']))
