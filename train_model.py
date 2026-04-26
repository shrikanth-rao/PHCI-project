import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv("phci_fake_dataset.csv")

WINDOW_SIZE = 100

def create_windows(signal, label):
    X, y = [], []
    for i in range(0, len(signal)-WINDOW_SIZE, WINDOW_SIZE):
        X.append(signal[i:i+WINDOW_SIZE])
        y.append(label)
    return X, y

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

# Build dataset
X, y = [], []

for label in [0,1,2]:
    sig = df[df['label']==label]['voltage_mv'].values
    windows, labels = create_windows(sig, label)
    
    for w in windows:
        X.append(extract_features(w))
        y.append(label)

# Train
model = RandomForestClassifier(n_estimators=150)
model.fit(X, y)

# Save model
joblib.dump(model, "phci_model.pkl")

print("✅ Model saved successfully")