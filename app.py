
import streamlit as st
import numpy as np
import joblib

# --- Custom CSS ---
st.markdown(
    """
    <style>
    body {
        background-image: linear-gradient(to bottom right, #e6f2ff, #ccf2ff);
    }
    .stButton>button {
        background-color: #00bfff;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
    }
    div[data-testid="stSlider"] > div > div > div[role="slider"] {
        background-color: #66ccff;
    }
    .css-1d391kg {
        background-color: #ffffff20;
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load the Trained Pipeline ---
model = joblib.load("Farms_Irrigation_System.pkl")

# --- Streamlit UI ---
st.title("Smart Sprinkler System")
st.subheader("Enter sensor values (0 to 1) to predict sprinkler status")

sensor_values = []
for i in range(20):
    val = st.slider(f"Sensor {i}", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    sensor_values.append(val)

# --- Predict on Button Click ---
if st.button("Predict Sprinklers"):
    input_array = np.array(sensor_values).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    st.markdown("### Prediction:")
    for i, status in enumerate(prediction):
        st.write(f"Sprinkler {i} (parcel_{i}): {'ON' if status == 1 else 'OFF'}")
