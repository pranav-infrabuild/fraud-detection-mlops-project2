"""
Drift detection with Evidently.
Compares old (training) data vs new (production) data.
"""
import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Load reference data (training data - the "normal" baseline)
print("Loading reference data...")
reference = pd.read_csv('data/processed/creditcard_processed.csv')

# Simulate new production data WITH drift (fraud patterns changed)
print("Simulating new production data with drift...")
production = reference.sample(n=500, random_state=1).copy()
# Shift some features to simulate changed patterns
for col in ['V1', 'V2', 'V3']:
    production[col] = production[col] + 1.5

# Build the drift report
print("Checking for drift...")
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference.drop('Class', axis=1),
           current_data=production.drop('Class', axis=1))

# Save as HTML so you can see it visually
report.save_html('reports/drift_report.html')
print("Done! Open reports/drift_report.html in your browser.")

# Print simple summary
result = report.as_dict()
drift_share = result['metrics'][0]['result']['share_of_drifted_columns']
print(f"\nDrifted features: {drift_share*100:.0f}%")
if drift_share > 0.3:
    print("ALERT: Too much drift! Model retraining recommended.")
else:
    print("OK: Drift is within acceptable range.")
