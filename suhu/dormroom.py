from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Fetch hourly weather data for Terengganu (past 3 months)
lat, lon = 5.3296, 103.1370
url = (
    f"https://api.open-meteo.com/v1/forecast?"
    f"latitude={lat}&longitude={lon}"
    f"&hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
    f"&past_days=90&forecast_days=1"
    f"&timezone=Asia%2FJakarta"
)
response = requests.get(url).json()

df = pd.DataFrame({
    "timestamp": pd.to_datetime(response["hourly"]["time"]),
    "temp_c": response["hourly"]["temperature_2m"],
    "humidity_pct": response["hourly"]["relative_humidity_2m"],
    "precipitation_mm": response["hourly"]["precipitation"],
    "wind_speed_kmh": response["hourly"]["wind_speed_10m"],
}).dropna()

df["hour"] = df["timestamp"].dt.hour

# Define consistent class labels in sorted order (matches sklearn's internal sorting)
CLASS_LABELS = [
    "1. Open Window (Breeze)",
    "2. Close Window (Rain/Bugs)",
    "3. Turn on Fan",
    "4. Close Window + AC",
]


# 2. Define Room Climate Control Scenarios
def determine_action(row):
  temp = row["temp_c"]
  humidity = row["humidity_pct"]
  rain = row["precipitation_mm"]
  wind = row["wind_speed_kmh"]

  # Scenario 4: Hot (>= 31.5°C) OR warm & muggy (>= 29.5°C and >= 80% humidity)
  if temp >= 31.5 or (temp >= 29.5 and humidity >= 80):
    return CLASS_LABELS[3]

  # Scenario 2: Raining (blocks water and flying insects) while not overly hot
  elif rain > 0:
    return CLASS_LABELS[1]

  # Scenario 1: Cool (< 29.0°C), dry, and a nice breeze (>= 8 km/h)
  elif temp < 29.0 and wind >= 8.0:
    return CLASS_LABELS[0]

  # Scenario 3: Dry, but either stagnant air (< 8 km/h) or moderately warm (29.0 - 31.4°C)
  else:
    return CLASS_LABELS[2]


df["action"] = df.apply(determine_action, axis=1)

# Add slight realistic variation (8% noise) so the tree learns natural decision boundaries
np.random.seed(42)
noise_idx = np.random.choice(df.index, size=int(0.08 * len(df)), replace=False)
df.loc[noise_idx, "action"] = np.random.choice(
    CLASS_LABELS, size=len(noise_idx)
)

# 3. Split Training (80%) & Testing (20%) Data
features = [
    "hour",
    "temp_c",
    "humidity_pct",
    "precipitation_mm",
    "wind_speed_kmh",
]
X = df[features]
y = df["action"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train Decision Tree (max_depth=4 to cleanly separate all 4 scenarios)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

print(
    "Model Accuracy on Test Data:"
    f" {accuracy_score(y_test, model.predict(X_test)):.1%}"
)

# 5. Visualize Decision Tree Flowchart
plt.figure(figsize=(22, 10))
plot_tree(
    model,
    feature_names=[
        "Hour",
        "Temp (°C)",
        "Humidity (%)",
        "Rain (mm)",
        "Wind (km/h)",
    ],
    class_names=CLASS_LABELS,
    filled=True,
    rounded=True,
    fontsize=9,
)
plt.title(
    "Room Climate Decision Tree: Window vs. Fan vs. AC",
    fontsize=15,
    fontweight="bold",
    pad=15,
)
plt.tight_layout()
plt.show()


# 6. Prediction Helper Function
def predict_action(
    temp_c: float,
    precipitation_mm: float,
    wind_speed_kmh: float = None,
    humidity_pct: float = None,
    hour: int = None,
):
  """Predicts the room window/fan/AC action based on weather inputs.

  Defaults to current hour and dataset median values for optional metrics.
  """
  if hour is None:
    hour = datetime.now().hour
  if humidity_pct is None:
    humidity_pct = df["humidity_pct"].median()
  if wind_speed_kmh is None:
    wind_speed_kmh = df["wind_speed_kmh"].median()

  input_data = pd.DataFrame([{
      "hour": hour,
      "temp_c": temp_c,
      "humidity_pct": humidity_pct,
      "precipitation_mm": precipitation_mm,
      "wind_speed_kmh": wind_speed_kmh,
  }])

  return model.predict(input_data)[0]


# --- Interactive CLI Test ---
if __name__ == "__main__":
  try:
    print("\n--- Room Climate Decision Check ---")
    input_temp = float(input("Enter current temperature (°C): "))
    input_rain = float(input("Enter precipitation (mm, enter 0 if dry): "))
    input_wind = float(input("Enter wind speed (km/h, e.g., 10 for breezy): "))
    input_humidity = float(input("Enter humidity (%, e.g., 75): "))

    decision = predict_action(
        temp_c=input_temp,
        precipitation_mm=input_rain,
        wind_speed_kmh=input_wind,
        humidity_pct=input_humidity,
    )
    print(f"\nRecommended Action: {decision}")

  except ValueError:
    print("Invalid input. Please enter numeric values.")


##-


import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import requests

# 1. Fetch hourly weather data for Kuala Terengganu, Malaysia (past 30 days)
lat, lon = 5.3296, 103.1370
url = (
    f"https://api.open-meteo.com/v1/forecast?"
    f"latitude={lat}&longitude={lon}"
    f"&hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
    f"&past_days=30&forecast_days=1"
    f"&timezone=Asia%2FKuala_Lumpur"
)
response = requests.get(url).json()

df = pd.DataFrame({
    "timestamp": pd.to_datetime(response["hourly"]["time"]),
    "temp_c": response["hourly"]["temperature_2m"],
    "humidity_pct": response["hourly"]["relative_humidity_2m"],
    "precipitation_mm": response["hourly"]["precipitation"],
    "wind_speed_kmh": response["hourly"]["wind_speed_10m"],
}).dropna()

# Filter strictly to the past 30 days up to the current hour
now = pd.Timestamp.now(tz="Asia/Kuala_Lumpur").tz_localize(None)
df = df[df["timestamp"] <= now].tail(30 * 24)

# 2. Plot all 4 variables across two synchronized subplots (30-Day View)
fig, (ax1, ax3) = plt.subplots(
    2,
    1,
    figsize=(16, 9),
    sharex=True,
    gridspec_kw={"height_ratios": [1.2, 1]},
)

# --- TOP PANEL: Temperature (°C) & Humidity (%) ---
color_temp = "tab:red"
ax1.set_ylabel("Temperature (°C)", color=color_temp, fontweight="bold")
line1 = ax1.plot(
    df["timestamp"],
    df["temp_c"],
    color=color_temp,
    linewidth=1.4,
    label="Temperature (°C)",
)
ax1.tick_params(axis="y", labelcolor=color_temp)
ax1.grid(True, linestyle="--", alpha=0.4)

ax2 = ax1.twinx()
color_humidity = "tab:blue"
ax2.set_ylabel("Humidity (%)", color=color_humidity, fontweight="bold")
line2 = ax2.plot(
    df["timestamp"],
    df["humidity_pct"],
    color=color_humidity,
    linewidth=1.2,
    alpha=0.75,
    label="Humidity (%)",
)
ax2.tick_params(axis="y", labelcolor=color_humidity)

# Combine legend for top panel
lines_top = line1 + line2
ax1.legend(
    lines_top,
    [l.get_label() for l in lines_top],
    loc="upper left",
    frameon=True,
)
ax1.set_title(
    "Kuala Terengganu Weather Variables — Last 30 Days (Hourly)",
    fontsize=14,
    fontweight="bold",
    pad=12,
)

# --- BOTTOM PANEL: Wind Speed (km/h) & Precipitation (mm) ---
color_wind = "tab:green"
ax3.set_ylabel("Wind Speed (km/h)", color=color_wind, fontweight="bold")
line3 = ax3.plot(
    df["timestamp"],
    df["wind_speed_kmh"],
    color=color_wind,
    linewidth=1.3,
    label="Wind Speed (km/h)",
)
# Add horizontal reference line for the 8 km/h "Nice Breeze" threshold
breeze_line = ax3.axhline(
    8.0,
    color="darkgreen",
    linestyle=":",
    linewidth=1.2,
    alpha=0.8,
    label="Breeze Threshold (8 km/h)",
)
ax3.tick_params(axis="y", labelcolor=color_wind)
ax3.grid(True, linestyle="--", alpha=0.4)

ax4 = ax3.twinx()
color_rain = "tab:cyan"
ax4.set_ylabel("Precipitation (mm)", color="teal", fontweight="bold")
bars = ax4.bar(
    df["timestamp"],
    df["precipitation_mm"],
    color=color_rain,
    width=0.05,
    alpha=0.7,
    label="Precipitation (mm)",
)
ax4.tick_params(axis="y", labelcolor="teal")
ax4.set_ylim(0, max(df["precipitation_mm"].max() * 1.2, 5))

# Combine legend for bottom panel
lines_bottom = line3 + [breeze_line, bars]
ax3.legend(
    lines_bottom,
    [l.get_label() for l in lines_bottom],
    loc="upper left",
    frameon=True,
)

# Format shared X-axis dates
ax3.set_xlabel("Date (MYT)", fontweight="bold")
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax3.xaxis.set_major_locator(mdates.DayLocator(interval=3))
plt.xticks(rotation=30)

fig.tight_layout()
plt.show()

