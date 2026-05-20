import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Load data
data = pd.read_csv("data/system_data.csv")

# Train model
model = IsolationForest(contamination=0.1)
model.fit(data)

# Save model
joblib.dump(model, "models/model.pkl")

print("Model trained and saved")