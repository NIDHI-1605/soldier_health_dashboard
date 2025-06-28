
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Soldier Health Dashboard", layout="centered")

st.title("🪖 Soldier Health Monitoring (Simulated)")

df = pd.read_csv("fake_WESAD_dataset.csv")

# UI to simulate vitals
heart_rate = st.slider("Heart Rate (BPM)", 50, 150, 80)
body_temp = st.slider("Body Temperature (°C)", 35, 42, 37)
eda = st.slider("EDA (µS)", 0.1, 10.0, 2.0)

# Simple preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(df[['heart_rate', 'temp', 'eda']])
y = df['risk']

model = RandomForestClassifier()
model.fit(X, y)

# Predict risk
input_data = scaler.transform([[heart_rate, body_temp, eda]])
prediction = model.predict(input_data)

if prediction[0] == 1:
    st.error("⚠️ ALERT: Soldier at Risk!")
else:
    st.success("✅ Soldier is Safe.")
