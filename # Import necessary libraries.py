# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
data = pd.read_csv('accident_data.csv')

# Define the dependent variable (DV) and independent variables (IVs)
X = data[['Road_Type', 'Weather_Conditions', 'Time_of_Day', 'Speed_Limit', 'Traffic_Signals', 'Pedestrian_Crossings', 'Speed_Bumps']]
y = data['Accident_Severity']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the model for future use
joblib.dump(model, 'accident_severity_model.pkl')

# Example of using the model to predict accident severity for a hypothetical set of independent variables
hypothetical_data = pd.DataFrame({
    'Road_Type': ['urban road'],
    'Weather_Conditions': ['clear'],
    'Time_of_Day': ['morning'],
    'Speed_Limit': [50],
    'Traffic_Signals': [1],
    'Pedestrian_Crossings': [0],
    'Speed_Bumps': [1]
})

# Make predictions
predicted_severity = model.predict(hypothetical_data)

print("Predicted Accident Severity:", predicted_severity[0])
