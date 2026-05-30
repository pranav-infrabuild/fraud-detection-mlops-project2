"""
Training script for Azure ML (without MLflow).
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json

# Load data
df = pd.read_csv('data/processed/creditcard_processed.csv')
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Train model
params = {'n_estimators': 100, 'max_depth': 10, 'class_weight': 'balanced', 'random_state': 42}
model = RandomForestClassifier(**params)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred)
}

print("=" * 50)
print("RESULTS")
for k, v in metrics.items():
    print(f"{k:12s}: {v:.4f}")
print("=" * 50)

# Save model
joblib.dump(model, 'outputs/fraud_model.pkl')
print("Model saved to outputs/")

# Save metrics for Azure ML
with open('outputs/metrics.json', 'w') as f:
    json.dump(metrics, f)
