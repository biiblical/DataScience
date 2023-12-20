import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load your dataset
df = pd.read_csv("C:/Users/tring/OneDrive/Desktop/manufacturing_Task_01.csv", sep=",")

# Assuming 'error_type' is your target variable
y_error_type = df['error_type']

# Features (X) - Include 'ionizationclass'
X_error_type = df.drop(['id', 'Quality', 'FluxCompensation', 'error', 'error_type', 'multideminsionality'], axis=1)

# Check for missing values in 'error_type'
print(df['error_type'].isnull().sum())

# Remove rows with missing or non-numeric values in 'error_type'
df_error_type = df.dropna(subset=['error_type'])

# Encode categorical labels if needed
# Assuming 'error_type' is categorical and needs encoding
label_encoder = LabelEncoder()  # Define LabelEncoder
df_error_type['error_type_encoded'] = label_encoder.fit_transform(df_error_type['error_type'])

# One-hot encode 'ionizationclass'
categorical_features = ['ionizationclass']
numeric_features = X_error_type.columns.difference(categorical_features)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', 'passthrough', numeric_features)
    ]
)

X_error_type_encoded = preprocessor.fit_transform(X_error_type)

# Split the data into training and testing sets
X_train_error_type, X_test_error_type, y_train_error_type, y_test_error_type = train_test_split(X_error_type_encoded, y_error_type, test_size=0.2, random_state=42)

# Create a Random Forest Classifier model
rf_model_error_type = RandomForestClassifier(random_state=42)

# Fit the model to the training data
rf_model_error_type.fit(X_train_error_type, y_train_error_type)

# Make predictions on the testing set
y_pred_error_type = rf_model_error_type.predict(X_test_error_type)

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test_error_type, y_pred_error_type)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for error_type Prediction (with ionizationclass)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print classification report
print("Classification Report:")
print(classification_report(y_test_error_type, y_pred_error_type, target_names=label_encoder.classes_))