# Example of using the model for prediction
new_data = [[2, 50, 60, 2, 1]]  # Wet road, visibility of 50 meters, speed limit of 60 km/h, 2 vehicles, daytime
predicted_severity = model.predict(new_data)

print(f'Predicted Accident Severity: {predicted_severity[0]:.2f}')
