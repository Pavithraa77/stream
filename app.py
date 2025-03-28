import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model

st.cache_resource.clear()
st.cache_data.clear()


# ------ Helper Functions ------
def create_dataset(data, time_step=1):
    dataX, dataY = [], []
    for i in range(len(data) - time_step - 1):
        dataX.append(data[i:(i + time_step), 0])
        dataY.append(data[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# ------ Data Loading & Preprocessing ------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_output.csv")
    df.rename(columns={"Column 1": "date", "Column 20": "usage"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.set_index("date", inplace=True)
    return df.dropna()

# ------ Model Loading ------
@st.cache_resource
def load_anomaly_model():
    return joblib.load("model.joblib")

@st.cache_resource
def load_lstm_model():
    with open("lstm_model1.pkl", "rb") as f:
        lstm_model = pickle.load(f)

    return lstm_model

# ------ Streamlit UI ------
st.title("💧 Water Usage Monitoring Dashboard")

data = load_data()

# ------ Anomaly Detection ------
st.header("🔍 Anomaly Detection")
anomaly_model = load_anomaly_model()
data["usage_scaled"] = (data["usage"] - data["usage"].mean()) / data["usage"].std()
data["anomaly_score"] = anomaly_model.predict(data[["usage_scaled"]])
data["anomaly"] = data["anomaly_score"].apply(lambda x: 1 if x == -1 else 0)

col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Recent Data")
    st.dataframe(data.tail())
with col2:
    st.subheader("⚠️ Detected Anomalies")
    st.dataframe(data[data["anomaly"] == 1].tail())

st.subheader("📈 Usage with Anomalies")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x=data.index, y=data["usage"], ax=ax, label="Normal Usage")
anomaly_points = data[data["anomaly"] == 1]
ax.scatter(anomaly_points.index, anomaly_points["usage"], color="red", label="Anomalies", marker="o")
ax.legend()
st.pyplot(fig)

# ------ LSTM Prediction ------
st.header("🔮 Usage Prediction (LSTM)")

quantity_data = data[["usage"]].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(quantity_data)

time_step = 10
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

lstm_model = load_lstm_model()
predictions = lstm_model.predict(X)

y_inv = scaler.inverse_transform(y.reshape(-1, 1))
predictions_inv = scaler.inverse_transform(predictions)

rmse = math.sqrt(mean_squared_error(y_inv, predictions_inv))
st.metric("Model RMSE", f"{rmse:.2f} units")

st.subheader("📉 Actual vs Predicted Usage")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index[time_step+1:], y_inv, label="Actual Usage")
ax.plot(data.index[time_step+1:], predictions_inv, label="Predicted Usage", alpha=0.7)
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
