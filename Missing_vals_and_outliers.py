import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def linear_regression():
    
    # Initialize model
    linear_reg = LinearRegression()

    # Train on training data
    linear_reg.fit(X_train_normal, y_train_normal)

    # Predict temperature for each city
    y_val_pred = linear_reg.predict(X_val_normal)

    # Evaluate performance
    mae = mean_absolute_error(y_val_normal, y_val_pred)
    rmse = np.sqrt(mean_squared_error(y_val_normal, y_val_pred))

    print(f"Linear Regression:\n MAE: {mae:.2f}\n RMSE: {rmse:.2f}")

def decision_tree_regressor():

    # Initialize Decision Tree
    tree_regressor = DecisionTreeRegressor(max_depth=10, random_state=42)

    # Train on training data
    tree_regressor.fit(X_train, y_train)

    # Predict
    y_val_pred_tree = tree_regressor.predict(X_val)

    # Evaluate
    mae_tree = mean_absolute_error(y_val, y_val_pred_tree)
    rmse_tree = np.sqrt(mean_squared_error(y_val, y_val_pred_tree))

    print(f"Decision Tree Regressor:\n MAE: {mae_tree:.2f}\n RMSE: {rmse_tree:.2f}")

def random_forest_regressor():

    # Initialize model
    rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

    # Train on training data
    rf_regressor.fit(X_train, y_train)

    # Predict
    y_val_pred_rf = rf_regressor.predict(X_val)

    # Evaluate
    mae_rf = mean_absolute_error(y_val, y_val_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_val, y_val_pred_rf))

    print(f"Random Forest Regressor:\n MAE: {mae_rf:.2f}\n RMSE: {rmse_rf:.2f}")

# Load the datasets
weather_data = pd.read_csv("Data/weather_prediction_dataset.csv")
labels = pd.read_csv("Data/weather_prediction_bbq_labels.csv")

#All the missing values and outiers has been dealt with already as stated in metadata


# Selecting numerical features (excluding date and month)
# Review: Maybe Month as time of the year could also be useful for predicting temperature, but we just need to one-hot encode
numerical_features = weather_data.drop(columns=["DATE", "MONTH"])

# Standardizing the features (zero mean, unit variance)
scaler = StandardScaler()
normalized_data = scaler.fit_transform(numerical_features)

# Convert back to DataFrame
normalized_df = pd.DataFrame(normalized_data, columns=numerical_features.columns)

# Display summary statistics after normalization
print("Normalized data frame:\n", normalized_df.describe())


# Reattach the DATE and MONTH columns for potential temporal analysis
normalized_df["DATE"] = weather_data["DATE"]
normalized_df["MONTH"] = weather_data["MONTH"]

# Define features (X) and target variable (y)
X_normal = normalized_df.drop(columns=["DATE", "MONTH"])  # Exclude date-related features
X = pd.DataFrame(numerical_features, columns=numerical_features.columns)
y = labels  # Assuming labels contain the target temperature

# Split into training (70%), validation (15%), and test (15%) sets
X_train_normal, X_temp_normal, y_train_normal, y_temp_normal = train_test_split(X_normal, y, test_size=0.30, random_state=42)
X_val_normal, X_test_normal, y_val_normal, y_test_normal = train_test_split(X_temp_normal, y_temp_normal, test_size=0.50, random_state=42)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# Display the shape of each set
print("Train // Validation // Test split (Samples, Features) \n", X_train.shape, X_val.shape, X_test.shape)

linear_regression()

decision_tree_regressor()

random_forest_regressor()