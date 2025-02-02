import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
file_path = "Data/weather_prediction_dataset.csv"
df = pd.read_csv(file_path)

# Convert date to datetime format
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
df.sort_values(by='DATE', inplace=True)

# Shift target variable TOURS_temp_mean by -1 to predict next day's temperature
df['TOURS_temp_mean_next_day'] = df['TOURS_temp_mean'].shift(-1)

# Drop last row (NaN in target due to shifting)
df.dropna(inplace=True)

# Select features (excluding DATE and target variable)
features = [col for col in df.columns if col not in ['DATE', 'TOURS_temp_mean_next_day']]
X = df[features]
y = df['TOURS_temp_mean_next_day']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# Predict next day's temperature using the latest available data
latest_data = X.iloc[-1].values.reshape(1, -1)
predicted_temp = model.predict(latest_data)
print(f"Predicted Temperature for the Next Day: {predicted_temp[0]:.2f}°C")