import streamlit as st
import pandas as pd
import joblib
import boto3
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
@@ -12,60 +10,56 @@
def load_model():
    return joblib.load("model.joblib")

# Load data from S3
# Load data from file
def load_data():
    bucket_name = "wateranomaly"
    file_name = "filtered_data.xlsx"
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket_name, Key=file_name)
    df = pd.read_excel(io.BytesIO(obj["Body"].read()))
    df = pd.read_excel("filtered_data.xlsx")
    return df

# Preprocess data
def preprocess_data(df):
    df.rename(columns={"Date": "date", "Water Used": "usage"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["date"])
    df = df[df["usage"] >= 0]
    df.set_index("date", inplace=True)
    return df

# Detect anomalies
def detect_anomalies(df, model):
    df = df.copy()
    df["usage_scaled"] = (df["usage"] - df["usage"].mean()) / df["usage"].std()
    df["anomaly_score"] = model.predict(df[["usage_scaled"]])
    df["anomaly"] = df["anomaly_score"].apply(lambda x: 1 if x == -1 else 0)
    return df

# Streamlit UI
st.title("💧 Water Usage Anomaly Detection")

# Load data & model
data = load_data()
data = preprocess_data(data)
model = load_model()
data = detect_anomalies(data, model)

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
sns.lineplot(x=data.index, y=data["usage"], ax=ax, label="Water Usage")
ax.scatter(data.index[data["anomaly"] == 1], data["usage"][data["anomaly"] == 1], color="red", label="Anomaly", marker="o")
ax.set_xlabel("Date")
ax.set_ylabel("Water Usage")
ax.set_title("Water Usage Anomaly Detection")
ax.legend()
st.pyplot(fig)
