import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.signal import find_peaks, welch

st.title("🌱 PHCI Dashboard")

# Load data + model
df = pd.read_csv("phci_fake_dataset.csv")
model = joblib.load("phci_model.pkl")

WINDOW_SIZE = 100

# Feature functions
def energy(w): return np.sum(w**2)
def zero_crossing_rate(w): return np.sum(np.diff(np.sign(w)) != 0)

def peak_features(w):
    peaks, _ = find_peaks(w)
    if len(peaks) == 0:
        return [0, 0]
    return [len(peaks), np.mean(w[peaks])]

def frequency_features(w):
    nperseg = min(64, len(w))
    freqs, psd = welch(w, nperseg=nperseg)
    return [freqs[np.argmax(psd)], np.sum(psd)]

def extract_features(w):
    return [
        np.mean(w), np.std(w), np.ptp(w),
        energy(w), zero_crossing_rate(w),
        *peak_features(w),
        *frequency_features(w)
    ]

def get_message(label):
    return ["🌿 Healthy", "💧 Water Needed", "🔥 Heat Stress"][label]

# Graph
st.subheader("📊 Signal")
st.line_chart(df['voltage_mv'][:1000])

# Prediction
conf = 0

if st.button("Run Live Prediction"):
    st.subheader("🧠 Results")

    for i in range(0, len(df)-WINDOW_SIZE, WINDOW_SIZE):
        window = df['voltage_mv'].values[i:i+WINDOW_SIZE]
        features = extract_features(window)

        pred = model.predict([features])[0]
        conf = max(model.predict_proba([features])[0])

        st.write(f"{get_message(pred)} (Confidence: {conf:.2f})")

# Progress bar safe
if conf > 0:
    st.progress(int(conf * 100))