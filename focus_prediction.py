import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load your data into a Pandas DataFrame (replace with your data source)
# Assumes you have a CSV file or similar source containing the data
# Features: 'pupil_dilation', 'point_of_gaze', 'blinks', 'eye_movement'
# Target: 'focus_loss_score'
data = pd.read_csv('student_focus_data.csv')

# Define features (X) and target (y)
X = data[['pupil_dilation', 'point_of_gaze', 'blinks', 'eye_movement']]
y = data['focus_loss_score']

# Preprocess data - Scale features (optional but useful in many models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the regression model (using Linear Regression here)
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Output the results
print(f'Mean Squared Error (MSE): {mse}')

# If you'd like to test with new data
new_data = np.array([[3.5, 4.2, 20, 0.8]])  # example values
new_data_scaled = scaler.transform(new_data)
focus_loss_pred = model.predict(new_data_scaled)

print(f'Predicted Focus Loss Score for new data: {focus_loss_pred[0]}')
