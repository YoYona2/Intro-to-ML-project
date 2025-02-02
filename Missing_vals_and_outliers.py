import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load the datasets
weather_data = pd.read_csv("Data/weather_prediction_dataset.csv")
labels = pd.read_csv("Data/weather_prediction_bbq_labels.csv")

# Check for missing values
missing_values = weather_data.isnull().sum()

# Identify outliers using the IQR method
Q1 = weather_data.quantile(0.25)
Q3 = weather_data.quantile(0.75)
IQR = Q3 - Q1
outliers = ((weather_data < (Q1 - 1.5 * IQR)) | (weather_data > (Q3 + 1.5 * IQR))).sum()

# Display missing values and outliers
print("Missing values:\n", missing_values)
print("\nOutliers:\n", outliers)


from sklearn.preprocessing import StandardScaler

# Selecting numerical features (excluding date and month)
numerical_features = weather_data.drop(columns=["DATE", "MONTH"])

# Standardizing the features (zero mean, unit variance)
scaler = StandardScaler()
normalized_data = scaler.fit_transform(numerical_features)

# Convert back to DataFrame
normalized_df = pd.DataFrame(normalized_data, columns=numerical_features.columns)

# Display summary statistics after normalization
print("Normalized data frame:\n", normalized_df.describe())

from sklearn.model_selection import train_test_split

# Reattach the DATE and MONTH columns for potential temporal analysis
normalized_df["DATE"] = weather_data["DATE"]
normalized_df["MONTH"] = weather_data["MONTH"]

# Define features (X) and target variable (y)
X = normalized_df.drop(columns=["DATE", "MONTH"])  # Exclude date-related features
y = labels  # Assuming labels contain the target temperature

# Split into training (70%), validation (15%), and test (15%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# Display the shape of each set
print("Train // Validation // Test split (Samples, Features) \n", X_train.shape, X_val.shape, X_test.shape)
