import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model

# ------ Helper Functions for LSTM ------
def create_dataset(data, time_step=1):
    dataX, dataY = [], []
    for i in range(len(data) - time_step - 1):
        dataX.append(data[i:(i + time_step), 0])
        dataY.append(data[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# ------ Data Loading & Preprocessing ------
@st.cache_data
def load_data():
    df = pd.read_excel("filtered_data.xlsx")
    df.rename(columns={"Date": "date", "Water Used": "usage"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["date"])
    df = df[df["usage"] >= 0]
    df.set_index("date", inplace=True)
    return df

# ------ Model Loading ------
@st.cache_resource
def load_anomaly_model():
    return joblib.load("model.joblib")

@st.cache_resource
def load_lstm_model():
    return load_model("lstm_model.h5")

# ------ Streamlit UI ------
st.title("💧 Water Usage Monitoring Dashboard")

# Load data
data = load_data()

# ------ Anomaly Detection Section ------
st.header("🔍 Anomaly Detection")
anomaly_model = load_anomaly_model()
data["usage_scaled"] = (data["usage"] - data["usage"].mean()) / data["usage"].std()
data["anomaly_score"] = anomaly_model.predict(data[["usage_scaled"]])
data["anomaly"] = data["anomaly_score"].apply(lambda x: 1 if x == -1 else 0)

# Display anomaly results
col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Recent Data")
    st.dataframe(data.tail())
with col2:
    st.subheader("⚠️ Detected Anomalies")
    st.dataframe(data[data["anomaly"] == 1].tail())

# Anomaly visualization
st.subheader("📈 Usage with Anomalies")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x=data.index, y=data["usage"], ax=ax, label="Normal Usage")
anomaly_points = data[data["anomaly"] == 1]
ax.scatter(anomaly_points.index, anomaly_points["usage"], 
           color="red", label="Anomalies", marker="o")
ax.set_xlabel("Date")
ax.set_ylabel("Water Usage")
ax.legend()
st.pyplot(fig)

# ------ LSTM Prediction Section ------
st.header("🔮 Usage Prediction (LSTM)")

# Prepare data for LSTM
quantity_data = data[["usage"]].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(quantity_data)

# Create sequences
time_step = 10
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Load LSTM model and make predictions
lstm_model = load_lstm_model()
predictions = lstm_model.predict(X)

# Inverse scaling
y_inv = scaler.inverse_transform(y.reshape(-1, 1))
predictions_inv = scaler.inverse_transform(predictions)

# Calculate RMSE
rmse = math.sqrt(mean_squared_error(y_inv, predictions_inv))
st.metric("Model RMSE", f"{rmse:.2f} units")

# Visualization
st.subheader("📉 Actual vs Predicted Usage")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index[time_step+1:], y_inv, label="Actual Usage")
ax.plot(data.index[time_step+1:], predictions_inv, 
        label="Predicted Usage", alpha=0.7)
ax.set_xlabel("Date")
ax.set_ylabel("Water Usage")
ax.legend()
st.pyplot(fig)

# ------ Data Statistics ------
st.header("📊 Data Insights")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Basic Statistics")
    st.write(data[["usage"]].describe())
with col2:
    st.subheader("Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data["usage"], kde=True, ax=ax)
    st.pyplot(fig)
