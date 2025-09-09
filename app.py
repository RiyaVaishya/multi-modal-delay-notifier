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

# =====================================================
# OPTION 1: Passenger Lookup (Ticket Authentication)
# =====================================================
st.header("📑 Passenger Lookup (from ticket data)")

if not ticket_df.empty:
    passenger_options = [
        f"{row['passenger_id']} - {row['name']}" 
        for _, row in ticket_df.iterrows()
    ]
    selected_passenger = st.selectbox("**Select Passenger**", passenger_options)
    selected_id = selected_passenger.split(" - ")[0]

    # Fetch Passenger Details
    record = ticket_df[ticket_df['passenger_id'] == selected_id].iloc[0]
    passenger_name = record['name']
    flight_id = record['flight_id']
    ferry_id = record['ferry_id']

    # Flight record
    flight_match = flight_df[flight_df['Flight'] == flight_id]
    if not flight_match.empty:
        flight = flight_match.iloc[0]

        dep_hour_flight = int(flight['ArrivalTime'].split(":")[0])
        origin_enc_flight = origin_encoder.transform([flight['Origin']])[0]
        dest_enc_flight = dest_encoder.transform([flight['Destination']])[0]
        flight_delay = model.predict([[dep_hour_flight, origin_enc_flight, dest_enc_flight]])[0]
    else:
        flight_delay = 0

    # Ferry record
    ferry_match = ferry_df[ferry_df['ferry_id'] == ferry_id]
    if not ferry_match.empty:
        ferry = ferry_match.iloc[0]

        dep_hour_ferry = int(ferry['departure_time'].split(":")[0])
        origin_enc_ferry = origin_encoder.transform([ferry['origin']])[0]
        dest_enc_ferry = dest_encoder.transform([ferry['destination']])[0]
        ferry_delay = model.predict([[dep_hour_ferry, origin_enc_ferry, dest_enc_ferry]])[0]
    else:
        ferry_delay = 0

    # Total delay & risk
    total_delay = flight_delay + ferry_delay
    if total_delay <= 10:
        risk_level = "🟢 Low Risk"
    elif total_delay <= 30:
        risk_level = "🟡 Medium Risk"
    else:
        risk_level = "🔴 High Risk"

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        if not flight_match.empty:
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
        if not ferry_match.empty:
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

    # Passenger Summary
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

# =====================================================
# OPTION 2: Manual Journey Input + Save in manual_entries.csv
# =====================================================
st.header("✈️ Enter Journey Details Manually")

with st.form("manual_input_form"):
    passenger_name = st.text_input("Enter Passenger Name")
    travel_mode = st.radio("Select Mode of Travel", ["Flight", "Ferry"])
    
    # Dropdowns for valid origins/destinations
    origin = st.selectbox("Select Origin", sorted(set(list(flight_df['Origin']) + list(ferry_df['origin']))))
    destination = st.selectbox("Select Destination", sorted(set(list(flight_df['Destination']) + list(ferry_df['destination']))))
    
    dep_time = st.text_input("Enter Departure Time (HH:MM format)")
    submit_button = st.form_submit_button("Predict Delay")

if submit_button:
    try:
        dep_hour = int(dep_time.split(":")[0])
        origin_enc = origin_encoder.transform([origin])[0]
        dest_enc = dest_encoder.transform([destination])[0]

        delay = model.predict([[dep_hour, origin_enc, dest_enc]])[0]

        # Risk level
        if delay <= 10:
            risk_level = "🟢 Low Risk"
        elif delay <= 30:
            risk_level = "🟡 Medium Risk"
        else:
            risk_level = "🔴 High Risk"

        # ---------------- Save manual entry into manual_entries.csv ----------------
        new_entry = {
            "passenger_name": passenger_name if passenger_name.strip() != "" else "Anonymous",
            "mode": travel_mode,
            "origin": origin,
            "destination": destination,
            "departure_time": dep_time,
            "predicted_delay": round(delay, 2),
            "risk_level": risk_level
        }

        try:
            manual_df = pd.read_csv("manual_entries.csv")
            manual_df = pd.concat([manual_df, pd.DataFrame([new_entry])], ignore_index=True)
        except FileNotFoundError:
            manual_df = pd.DataFrame([new_entry])

        manual_df.to_csv("manual_entries.csv", index=False)
        st.success("✅ Manual entry saved to manual_entries.csv")

        # ---------------- Show prediction result ----------------
        st.success(f"Predicted Delay for {travel_mode}: {delay:.2f} minutes")
        st.info(f"🚨 Risk Level: {risk_level}")

    except Exception as e:
        st.error("⚠️ Something went wrong while predicting. Please check input format.")
    