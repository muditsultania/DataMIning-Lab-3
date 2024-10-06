import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv('student_attendance_performance.csv')

# Features and target
X = df[['Attendance_Percentage', 'Assignment_Score', 'Exam_Score', 'Class_Participation']]
y = df['Is_At_Risk']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Implement KNN (using k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Display students and their classifications
results = X_test.copy()
results['Actual_Risk'] = y_test
results['Predicted_Risk'] = y_pred
print("\nStudents and their classification:\n", results)
print("Mudit Sultania   21BBS0232")

