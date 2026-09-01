import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Delivery Time Predictor", page_icon="🛵")

# ---- Load model bundle ----
@st.cache_resource
def load_bundle():
    with open("model_bundle.pkl", "rb") as f:
        return pickle.load(f)

try:
    bundle = load_bundle()

    model = bundle["model"]
    scaler = bundle["scaler"]

    vehicle_classes = ["Motorcycle", "Scooter", "Electric"]

    feature_order = [
        "Delivery_person_Age",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
        "Type_of_vehicle"
    ]

except FileNotFoundError:
    st.error("model_bundle.pkl not found.")
    st.stop()

st.title("🛵 Delivery Time Predictor")
st.write("Estimate how many minutes a food delivery will take.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Rider")
    age = st.number_input("Delivery person age", min_value=15, max_value=70, value=30)
    vehicle = st.selectbox("Vehicle type", vehicle_classes)

with col2:
    st.subheader("Vehicle encoding check")
    st.caption(f"Encoded as: {vehicle_classes.index(vehicle)}")

st.subheader("Restaurant location")
rc1, rc2 = st.columns(2)
with rc1:
    rest_lat = st.number_input("Restaurant latitude", value=22.745049, format="%.6f")
with rc2:
    rest_lon = st.number_input("Restaurant longitude", value=75.892471, format="%.6f")

st.subheader("Delivery location")
dc1, dc2 = st.columns(2)
with dc1:
    del_lat = st.number_input("Delivery latitude", value=22.765049, format="%.6f")
with dc2:
    del_lon = st.number_input("Delivery longitude", value=75.912471, format="%.6f")

if st.button("Predict delivery time", type="primary"):
    vehicle_encoded = vehicle_classes.index(vehicle)

    # Build a row in the EXACT column order the model was trained on
    input_dict = {
        "Delivery_person_Age": age,
        "Restaurant_latitude": rest_lat,
        "Restaurant_longitude": rest_lon,
        "Delivery_location_latitude": del_lat,
        "Delivery_location_longitude": del_lon,
        "Type_of_vehicle": vehicle_encoded,
    }
    input_df = pd.DataFrame([input_dict])[feature_order]

    # Apply the SAME scaler used in training -- this is the step
    # the original notebook skipped for its single prediction.
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    prediction = max(prediction, 0)  # time can't be negative

    st.success(f"Estimated delivery time: **{prediction:.1f} minutes**")

st.divider()
st.caption(
    "Model: Linear Regression trained on rider age, pickup/dropoff "
    "coordinates, and vehicle type. Note: this model's R² score on the "
    "test set is low (~9%), so treat predictions as rough estimates."
)
