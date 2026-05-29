"""
Generate synthetic credit card fraud dataset.
"""
import pandas as pd
import numpy as np

np.random.seed(42)

def generate_fraud_dataset(n_samples=10000, fraud_ratio=0.02):
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Legitimate transactions
    legit_amounts = np.random.gamma(shape=2, scale=50, size=n_legit)
    legit_times = np.random.uniform(0, 172800, n_legit)
    legit_features = np.random.randn(n_legit, 10)

    # Fraudulent transactions (different patterns)
    fraud_amounts = np.random.gamma(shape=5, scale=100, size=n_fraud)
    fraud_times = np.random.uniform(0, 172800, n_fraud)
    fraud_features = np.random.randn(n_fraud, 10) * 2 + 1

    amounts = np.concatenate([legit_amounts, fraud_amounts])
    times = np.concatenate([legit_times, fraud_times])
    features = np.vstack([legit_features, fraud_features])
    labels = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

    data = {'Time': times, 'Amount': amounts, 'Class': labels}
    for i in range(10):
        data[f'V{i+1}'] = features[:, i]

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

if __name__ == "__main__":
    print("Generating synthetic fraud dataset...")
    df = generate_fraud_dataset()
    df.to_csv('data/raw/creditcard.csv', index=False)
    print(f"Done! Total: {len(df)}, Fraud: {df['Class'].sum():.0f}")