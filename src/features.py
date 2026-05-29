"""
Feature engineering for fraud detection.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def engineer_features(df):
    df = df.copy()
    # Time-based features
    df['Hour'] = (df['Time'] / 3600) % 24
    df['Is_Night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
    # Amount-based features
    df['Log_Amount'] = np.log1p(df['Amount'])
    df['Amount_Bin'] = pd.cut(df['Amount'], bins=[0, 50, 100, 500, 100000],
                              labels=[0, 1, 2, 3]).fillna(0).astype(int)
    df['High_Risk'] = ((df['Amount'] > 500) & (df['Is_Night'] == 1)).astype(int)
    return df

def prepare_data(input_path, output_path):
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    print(f"Original shape: {df.shape}, Fraud ratio: {df['Class'].mean()*100:.2f}%")

    df = engineer_features(df)

    X = df.drop('Class', axis=1)
    y = df['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved to models/scaler.pkl")

    df_final = pd.DataFrame(X_scaled, columns=X.columns)
    df_final['Class'] = y.values
    df_final.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    prepare_data('data/raw/creditcard.csv', 'data/processed/creditcard_processed.csv')