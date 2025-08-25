import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load datasets
flight_df = pd.read_csv("flight_data.csv")
ferry_df = pd.read_csv("ferry_data.csv")

# Standardize column names
flight_df = flight_df.rename(columns={
    'ArrivalTime': 'departure_time',
    'ExpectedDelayMin': 'delay_minutes',
    'Flight': 'id'
})

ferry_df = ferry_df.rename(columns={
    'ferry_id': 'id'
})

# Merge datasets for training
combined_df = pd.concat([
    flight_df[['departure_time', 'Origin', 'Destination', 'delay_minutes']],
    ferry_df[['departure_time', 'origin', 'destination', 'delay_minutes']].rename(
        columns={'origin': 'Origin', 'destination': 'Destination'})
], ignore_index=True)

# Feature engineering
combined_df['dep_hour'] = combined_df['departure_time'].apply(lambda x: int(x.split(":")[0]))

origin_encoder = LabelEncoder()
dest_encoder = LabelEncoder()
combined_df['origin_enc'] = origin_encoder.fit_transform(combined_df['Origin'])
combined_df['dest_enc'] = dest_encoder.fit_transform(combined_df['Destination'])

# Split data
X = combined_df[['dep_hour', 'origin_enc', 'dest_enc']]
y = combined_df['delay_minutes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model & encoders
joblib.dump(model, "delay_predictor.pkl")
joblib.dump(origin_encoder, "origin_encoder.pkl")
joblib.dump(dest_encoder, "dest_encoder.pkl")

print("✅ Model trained and saved!")