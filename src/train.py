"""
Model training with MLflow experiment tracking.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
import mlflow
import mlflow.sklearn
import joblib
from datetime import datetime

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("fraud-detection")

def train_model(data_path, params):
    with mlflow.start_run(run_name=f"rf-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        df = pd.read_csv(data_path)
        X = df.drop('Class', axis=1)
        y = df['Class']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_params(params)

        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        mlflow.log_metrics(metrics)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        joblib.dump(model, 'models/fraud_model.pkl')
        mlflow.sklearn.log_model(model, "model")

        print("=" * 50)
        print("RESULTS")
        for k, v in metrics.items():
            print(f"{k:12s}: {v:.4f}")
        print(f"Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
        print("=" * 50)

        mlflow.set_tags({"model_type": "RandomForest", "project": "fraud-detection"})

if __name__ == "__main__":
    # Experiment 1: Baseline
    print("Experiment 1: Baseline")
    train_model('data/processed/creditcard_processed.csv',
                {'n_estimators': 100, 'max_depth': 10, 'class_weight': 'balanced'})

    # Experiment 2: More trees
    print("\nExperiment 2: More capacity")
    train_model('data/processed/creditcard_processed.csv',
                {'n_estimators': 200, 'max_depth': 15, 'class_weight': 'balanced'})

    print("\nDone! Run 'mlflow ui' to see experiments.")