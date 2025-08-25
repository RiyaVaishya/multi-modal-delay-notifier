import streamlit as st
import pandas as pd
import joblib

# ------------------ Load ML models ------------------
model = joblib.load("delay_predictor.pkl")
origin_encoder = joblib.load("origin_encoder.pkl")
dest_encoder = joblib.load("dest_encoder.pkl")

# ------------------ Load data ------------------
ticket_df = pd.read_csv("ticket_data.csv")
flight_df = pd.read_csv("flight_data.csv")
ferry_df = pd.read_csv("ferry_data.csv")

# ------------------ Page config ------------------
st.set_page_config(page_title="Multi-Modal Delay Notifier", layout="wide", page_icon="🛫")

# ------------------ Custom CSS ------------------
st.markdown(
    """
    <style>
        body { background-color: #0e1117; }
        .title { font-size: 2.5em; font-weight: bold; text-align: center; color: white; }
        .subtitle { font-size: 1.2em; text-align: center; color: #9ca3af; margin-bottom: 20px; }
        .section-title { font-size: 1.4em; font-weight: bold; margin-top: 15px; color: white; }
        .info-card {
            background-color: #1f2937;
            padding: 20px;
            border-radius: 10px;
            color: white;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.5);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Title & Subtitle ------------------
st.markdown("<div class='title'>🛫🚢 Multi-Modal Delay Notifier System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predicting combined flight + ferry delays for passengers using Machine Learning</div>", unsafe_allow_html=True)
st.markdown("---")

# ------------------ Dropdown with ID + Name ------------------
passenger_options = [
    f"{row['passenger_id']} - {row['name']}" 
    for _, row in ticket_df.iterrows()
]
selected_passenger = st.selectbox("**Select Passenger**", passenger_options)
selected_id = selected_passenger.split(" - ")[0]

# ------------------ Fetch Passenger Details ------------------
record = ticket_df[ticket_df['passenger_id'] == selected_id].iloc[0]
passenger_name = record['name']
flight_id = record['flight_id']
ferry_id = record['ferry_id']

# Flight record
flight_match = flight_df[flight_df['Flight'] == flight_id]
if flight_match.empty:
    st.error(f"No flight found with ID: {flight_id}")
    st.stop()
flight = flight_match.iloc[0]

# Ferry record
ferry_match = ferry_df[ferry_df['ferry_id'] == ferry_id]
if ferry_match.empty:
    st.error(f"No ferry found with ID: {ferry_id}")
    st.stop()
ferry = ferry_match.iloc[0]

# ------------------ Predictions ------------------
# Flight delay
dep_hour_flight = int(flight['ArrivalTime'].split(":")[0])
origin_enc_flight = origin_encoder.transform([flight['Origin']])[0]
dest_enc_flight = dest_encoder.transform([flight['Destination']])[0]
flight_delay = model.predict([[dep_hour_flight, origin_enc_flight, dest_enc_flight]])[0]

# Ferry delay
dep_hour_ferry = int(ferry['departure_time'].split(":")[0])
origin_enc_ferry = origin_encoder.transform([ferry['origin']])[0]
dest_enc_ferry = dest_encoder.transform([ferry['destination']])[0]
ferry_delay = model.predict([[dep_hour_ferry, origin_enc_ferry, dest_enc_ferry]])[0]

# Total delay
total_delay = flight_delay + ferry_delay

# Risk level
if total_delay <= 10:
    risk_level = "🟢 Low Risk"
elif total_delay <= 30:
    risk_level = "🟡 Medium Risk"
else:
    risk_level = "🔴 High Risk"

# ------------------ Layout ------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='section-title'>🛫 Flight Info</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='info-card'>
        <b>Flight ID:</b> {flight_id}<br>
        <b>Route:</b> {flight['Origin']} → {flight['Destination']}<br>
        <b>Scheduled Time:</b> {flight['ArrivalTime']}<br>
        <b>Predicted Delay:</b> {flight_delay:.2f} min
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown("<div class='section-title'>🚢 Ferry Info</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='info-card'>
        <b>Ferry ID:</b> {ferry_id}<br>
        <b>Route:</b> {ferry['origin']} → {ferry['destination']}<br>
        <b>Scheduled Time:</b> {ferry['departure_time']}<br>
        <b>Predicted Delay:</b> {ferry_delay:.2f} min
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------ Passenger Summary ------------------
st.markdown("---")
st.markdown(
    f"""
    <div class='info-card' style='text-align:center; font-size:1.2em;'>
        👤 <b>Passenger:</b> {passenger_name} <br>
        🧮 <b>Total Predicted Delay:</b> {total_delay:.2f} minutes <br>
        🚨 <b>Risk Level:</b> {risk_level}
    </div>
    """,
    unsafe_allow_html=True
)