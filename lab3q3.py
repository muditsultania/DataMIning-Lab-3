import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv('fruit_dataset.csv')

# Encode categorical variables (Fruit, Color) to numerical values
le_fruit = LabelEncoder()
le_color = LabelEncoder()
df['Fruit'] = le_fruit.fit_transform(df['Fruit'])
df['Color'] = le_color.fit_transform(df['Color'])

# Features (Weight, Color, Sweetness_Level)
X = df[['Weight (g)', 'Color', 'Sweetness_Level (1-10)']]

# Target (Fruit type)
y = df['Fruit']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = rf_classifier.predict(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Display actual vs predicted fruit labels
actual_fruit = le_fruit.inverse_transform(y_test)
predicted_fruit = le_fruit.inverse_transform(y_pred)

results = pd.DataFrame({
    'Actual Fruit': actual_fruit,
    'Predicted Fruit': predicted_fruit
})

print(results)
print("Mudit Sultania 21BBS0232")
