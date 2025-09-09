import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
flight_df = pd.read_csv("flight_data.csv")
ferry_df = pd.read_csv("ferry_data.csv")

# ---------------- 2.6 Data Analysis ----------------
# Average delay per origin city (Flights)
avg_flight_delay = flight_df.groupby("Origin")["ExpectedDelayMin"].mean()

plt.figure(figsize=(8,5))
avg_flight_delay.plot(kind="bar", color="blue", edgecolor="black")
plt.title("Average Flight Delay by Origin City")
plt.ylabel("Delay (minutes)")
plt.xlabel("Origin City")
plt.tight_layout()
plt.show()

# ---------------- 2.7 Data Visualization ----------------
# Distribution of flight delays
plt.figure(figsize=(8,5))
plt.hist(flight_df["ExpectedDelayMin"], bins=6, edgecolor="black", color="purple")
plt.title("Distribution of Flight Delays")
plt.xlabel("Delay (minutes)")
plt.ylabel("Number of Flights")
plt.tight_layout()
plt.show()
