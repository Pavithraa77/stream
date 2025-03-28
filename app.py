import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# Load Isolation Forest model
@st.cache_resource
def load_anomaly_model():
    return joblib.load("model.joblib")

# Load LSTM model
@st.cache_resource
def load_lstm_model():
    return load_model("lstm_model.h5")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_excel("filtered_data.xlsx")
    df.rename(columns={'Date': 'date', 'Water Used': 'quantity'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], errors="coerce")
    df.set_index('date', inplace=True)
    df = df.dropna()
    return df

# Preprocess data
def preprocess_data(df):
    df = df[df["quantity"] >= 0]
    return df

# Detect anomalies
def detect_anomalies(df, model):
    df = df.copy()
    df["quantity_scaled"] = (df["quantity"] - df["quantity"].mean()) / df["quantity"].std()
    df["anomaly_score"] = model.predict(df[["quantity_scaled"]])
    df["anomaly"] = df["anomaly_score"].apply(lambda x: 1 if x == -1 else 0)
    return df

# Predict using LSTM model
def lstm_predict(df, model):
    df = df.copy()
    data = df["quantity"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    time_step = 10
    X_test = []
    for i in range(len(scaled_data) - time_step - 1):
        X_test.append(scaled_data[i:(i + time_step), 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    df = df.iloc[time_step + 1:]
    df["predicted_quantity"] = predictions
    return df

# Streamlit UI
st.title("💧 Water Usage Anomaly Detection & Prediction")

# Load data & models
data = load_data()
data = preprocess_data(data)
anomaly_model = load_anomaly_model()
lstm_model = load_lstm_model()
data = detect_anomalies(data, anomaly_model)
data = lstm_predict(data, lstm_model)

# Show raw data
st.subheader("📊 Raw Data")
st.dataframe(data)

# Show statistics
st.subheader("📈 Data Insights")
st.write(data.describe())

# Show anomalies
st.subheader("⚠️ Detected Anomalies")
st.dataframe(data[data["anomaly"] == 1])

# Plot anomalies
st.subheader("📉 Water Usage with Anomalies")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x=data.index, y=data["quantity"], ax=ax, label="Water Usage")
ax.scatter(data.index[data["anomaly"] == 1], data["quantity"][data["anomaly"] == 1], color="red", label="Anomaly", marker="o")
ax.set_xlabel("Date")
ax.set_ylabel("Water Usage")
ax.set_title("Water Usage Anomaly Detection")
ax.legend()
st.pyplot(fig)

# Plot LSTM Predictions
st.subheader("📊 LSTM Predictions vs Actual Values")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x=data.index, y=data["quantity"], ax=ax, label="Actual Usage")
sns.lineplot(x=data.index, y=data["predicted_quantity"], ax=ax, label="Predicted Usage")
ax.set_xlabel("Date")
ax.set_ylabel("Water Usage")
ax.set_title("LSTM Model Predictions")
ax.legend()
st.pyplot(fig)
