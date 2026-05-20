from flask import Flask, jsonify, render_template
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
import psutil
import os
import joblib
import pandas as pd

app = Flask(__name__)

# Load ML model safely
model = joblib.load("models/model.pkl")

# Prometheus metrics
cpu_metric = Gauge('cpu_usage', 'CPU Usage')
memory_metric = Gauge('memory_usage', 'Memory Usage')
anomaly_metric = Gauge('anomaly_status', 'Anomaly Status')

# Prediction function (FIXED WARNING ISSUE)
def predict(cpu, memory):
    data = pd.DataFrame([[cpu, memory]], columns=["cpu", "memory"])
    result = model.predict(data)
    return "🚨 Anomaly" if result[0] == -1 else "Normal"

@app.route('/')
def home():
    return "AI Monitoring Running 🚀"

@app.route('/metrics')
def metrics():
    cpu = psutil.cpu_percent()
    memory = psutil.virtual_memory().percent

    result = predict(cpu, memory)

    cpu_metric.set(cpu)
    memory_metric.set(memory)
    anomaly_metric.set(1 if result != "Normal" else 0)

    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/data')
def data():
    cpu = psutil.cpu_percent()
    memory = psutil.virtual_memory().percent
    result = predict(cpu, memory)

    return jsonify({
        "cpu": cpu,
        "memory": memory,
        "status": result
    })

@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")

# 🔥 IMPORTANT FOR RENDER
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)