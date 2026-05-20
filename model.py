import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

def predict(cpu, memory):
    # Fix: use DataFrame with column names
    data = pd.DataFrame([[cpu, memory]], columns=["cpu", "memory"])

    result = model.predict(data)
    return "🚨 Anomaly" if result[0] == -1 else "Normal"