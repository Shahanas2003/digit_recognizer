import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.datasets import mnist

st.title("🖊️ Handwritten Digit Classifier (Logistic Regression)")

# Load data
@st.cache_data
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X = np.concatenate((x_train, x_test), axis=0).reshape(-1, 28*28)
    y = np.concatenate((y_train, y_test), axis=0)
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = load_data()

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

st.write(f"🎯 Accuracy: {acc:.4f}")

# Confusion Matrix
fig, ax = plt.subplots()
sns.heatmap(
    pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']),
    annot=True, fmt='d', cmap='Blues', ax=ax
)
st.pyplot(fig)
