import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# ------------------ Load datasets ------------------
flight_df = pd.read_csv("flight_data.csv")
ferry_df = pd.read_csv("ferry_data.csv")

# ------------------ Standardize column names ------------------
flight_df = flight_df.rename(columns={
    'ArrivalTime': 'departure_time',
    'ExpectedDelayMin': 'delay_minutes',
    'Flight': 'id'
})

ferry_df = ferry_df.rename(columns={
    'ferry_id': 'id'
})

# ------------------ Merge datasets for training ------------------
combined_df = pd.concat([
    flight_df[['departure_time', 'Origin', 'Destination', 'delay_minutes']],
    ferry_df[['departure_time', 'origin', 'destination', 'delay_minutes']].rename(
        columns={'origin': 'Origin', 'destination': 'Destination'})
], ignore_index=True)

# ------------------ Feature Engineering ------------------
combined_df['dep_hour'] = combined_df['departure_time'].apply(lambda x: int(x.split(":")[0]))

# ------------------ Add extra possible routes (for manual input flexibility) ------------------
extra_data = pd.DataFrame({
    "Origin": ["Mumbai", "Delhi", "Goa", "Chennai", "Bangalore"],
    "Destination": ["Goa", "Delhi", "Mumbai", "Kolkata", "Hyderabad"],
    "departure_time": ["10:00", "12:00", "14:00", "16:00", "18:00"],
    "delay_minutes": [15, 25, 5, 20, 30]
})
extra_data['dep_hour'] = extra_data['departure_time'].apply(lambda x: int(x.split(":")[0]))

# Append extra routes
combined_df = pd.concat([combined_df, extra_data], ignore_index=True)

# ------------------ Encode categorical variables ------------------
origin_encoder = LabelEncoder()
dest_encoder = LabelEncoder()
combined_df['origin_enc'] = origin_encoder.fit_transform(combined_df['Origin'])
combined_df['dest_enc'] = dest_encoder.fit_transform(combined_df['Destination'])

# ------------------ Split Data ------------------
X = combined_df[['dep_hour', 'origin_enc', 'dest_enc']]
y = combined_df['delay_minutes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ Train Model ------------------
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# ------------------ Save Model & Encoders ------------------
joblib.dump(model, "delay_predictor.pkl")
joblib.dump(origin_encoder, "origin_encoder.pkl")
joblib.dump(dest_encoder, "dest_encoder.pkl")

print("✅ Model trained with extended routes and saved successfully!")