import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('demat_account_data.csv')

# Convert Month column to datetime format
data['Month'] = pd.to_datetime(data['Month'])

# Feature and target
data['Months_Since_Start'] = (data['Month'] - data['Month'].min()).dt.days // 30
X = data[['Months_Since_Start']]
y = data['DEMAT_Accounts']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future DEMAT accounts for January 2025 (approx. 60 months from the start)
months_since_start_jan2025 = (pd.to_datetime('2025-01') - data['Month'].min()).days // 30
prediction = model.predict([[months_since_start_jan2025]])

print(f"Predicted DEMAT accounts for January 2025: {prediction[0]:.2f} million")

# Plot the regression line
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title('DEMAT Account Growth Prediction')
plt.xlabel('Months Since Start')
plt.ylabel('DEMAT Accounts (in millions)')
plt.show()

print("Mudit Sultania 21BBS0232")
