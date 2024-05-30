import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
data = pd.read_csv('medical_data.csv')

# Preprocess the data
data['sex'] = data['sex'].apply(lambda x: 1 if x == 'M' else 0)
data['symptoms'] = data['symptoms'].apply(lambda x: len(x.split('|')))  # Simplistic symptom encoding
data['patient_history'] = data['patient_history'].apply(lambda x: 1 if x == 'smoker' else 0)

# Split data into features and target
X = data.drop('disease', axis=1)
y = data['disease']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'disease_prediction_model.pkl')
