import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("heart.csv")

# Encode categorical columns
data['sex'] = data['sex'].map({'Male': 1, 'Female': 0})
data['exercise_induced_angina'] = data['exercise_induced_angina'].map({'Yes': 1, 'No': 0})

# Select important features
selected_features = ["age", "sex", "cholestoral", "Max_heart_rate", "exercise_induced_angina", "oldpeak"]
X = data[selected_features]
y = data["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Calculate accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))

# Save model, scaler, and accuracy
pickle.dump(model, open("heart_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(accuracy, open("accuracy.pkl", "wb"))

print(f"✅ Model retrained! Accuracy: {accuracy:.2f}")