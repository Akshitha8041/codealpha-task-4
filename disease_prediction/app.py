from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the model
model = joblib.load('disease_prediction_model.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Disease Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Convert input data to DataFrame
    df = pd.DataFrame([data])
    df['sex'] = df['sex'].apply(lambda x: 1 if x.lower() == 'm' else 0)
    df['symptoms'] = df['symptoms'].apply(lambda x: len(x.split('|')))
    df['patient_history'] = df['patient_history'].apply(lambda x: 1 if x.lower() == 'smoker' else 0)

    # Make prediction
    prediction = model.predict(df)
    output = int(prediction[0])
    
    return jsonify({'prediction': output})

if __name__ == '__main__':
    app.run(debug=True)
