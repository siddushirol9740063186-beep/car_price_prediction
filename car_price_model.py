import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv("car_data.csv")

# Convert categorical → numeric
data['Fuel_Type'] = data['Fuel_Type'].map({'Petrol': 2, 'Diesel': 1, 'CNG': 0})
data['Seller_Type'] = data['Seller_Type'].map({'Dealer': 0, 'Individual': 1})
data['Transmission'] = data['Transmission'].map({'Manual': 1, 'Automatic': 0})

# Feature Engineering
data['Car_Age'] = 2025 - data['Year']

# Drop unused columns
data = data.drop(['Car_Name', 'Year'], axis=1)

# Split data
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("model.pkl created successfully")
