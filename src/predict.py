"""
Flask API for fraud detection inference.
"""
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

print("Loading model and scaler...")
model = joblib.load('models/fraud_model.pkl')
scaler = joblib.load('models/scaler.pkl')
print("Model loaded successfully!")

def engineer_features(data):
    df = pd.DataFrame([data])
    df['Hour'] = (df['Time'] / 3600) % 24
    df['Is_Night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
    df['Log_Amount'] = np.log1p(df['Amount'])
    df['Amount_Bin'] = pd.cut(df['Amount'], bins=[0, 50, 100, 500, 100000],
                              labels=[0, 1, 2, 3]).fillna(0).astype(int)
    df['High_Risk'] = ((df['Amount'] > 500) & (df['Is_Night'] == 1)).astype(int)
    cols = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8',
            'V9', 'V10', 'Hour', 'Is_Night', 'Log_Amount', 'Amount_Bin', 'High_Risk']
    return df[cols]

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        df = engineer_features(data)
        X = scaler.transform(df)
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        result = {
            'prediction': int(prediction),
            'fraud_probability': round(float(probability[1]), 4),
            'risk_level': 'HIGH' if probability[1] > 0.7 else 'MEDIUM' if probability[1] > 0.3 else 'LOW',
            'timestamp': datetime.now().isoformat()
        }
        print(f"Prediction: {result['prediction']} (prob: {result['fraud_probability']})")
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Fraud Detection API on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)