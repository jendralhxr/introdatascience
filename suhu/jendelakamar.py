import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Tarik data cuaca Surabaya per jam (14 hari terakhir)
lat, lon = -7.2575, 112.7521
url = (
    f"https://api.open-meteo.com/v1/forecast?"
    f"latitude={lat}&longitude={lon}"
    f"&hourly=temperature_2m,relative_humidity_2m,precipitation"
    f"&past_days=14&forecast_days=1"
    f"&timezone=Asia%2FJakarta"
)
res = requests.get(url).json()

df = pd.DataFrame({
    "waktu": pd.to_datetime(res["hourly"]["time"]),
    "suhu_c": res["hourly"]["temperature_2m"],
    "kelembapan_persen": res["hourly"]["relative_humidity_2m"],
    "hujan_mm": res["hourly"]["precipitation"],
}).dropna()

df["jam"] = df["waktu"].dt.hour


# 2. Buat Label Keputusan Anak Kos Surabaya
# (1) Buka Jendela: Suhu nyaman (< 29.5°C) dan tidak hujan
# (2) Tutup Jendela, Nyalakan AC: Suhu gerah (>= 29.5°C) atau sedang hujan
# (3) Ngafe: Panas menyengat (>= 32.5°C) di jam siang-sore (11.00 - 17.00 WIB) & tidak hujan
def tentukan_keputusan(row):
  # 1. Buka Jendela: Suhu sejuk (< 29.5°C) dan tidak hujan
  if row["suhu_c"] < 29.5 and row["hujan_mm"] == 0:
    return "1. Buka Jendela"
  # 2. Tutup Jendela + AC: Hujan, atau suhu hangat-panas (29.5°C - 32.4°C), atau panas di luar jam ngafe
  elif (
      row["hujan_mm"] > 0
      or row["suhu_c"] < 32.5
      or not (11 <= row["jam"] <= 17)
  ):
    return "2. Tutup Jendela + AC"
  # 3. Ngafe: Sisa kondisi (Suhu >= 32.5°C, tidak hujan, dan jam 11.00 - 17.00 WIB)
  else:
    return "3. Ngafe"

df["keputusan"] = df.apply(tentukan_keputusan, axis=1)

# Tambahkan sedikit variasi realistis (10% noise) agar pohon keputusan belajar pola alami
noise_idx = np.random.choice(df.index, size=int(0.10 * len(df)), replace=False)
df.loc[noise_idx, "keputusan"] = np.random.choice(
    ["1. Buka Jendela", "2. Tutup Jendela + AC", "3. Ngafe"],
    size=len(noise_idx),
)

# 3. Bagi Data Latih (80%) & Data Uji (20%)
X = df[["jam", "suhu_c", "kelembapan_persen", "hujan_mm"]]
y = df["keputusan"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Latih Decision Tree (max_depth=3 agar mudah dibaca mahasiswa)
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

print(
    f"Akurasi Model pada Data Uji: {accuracy_score(y_test, model.predict(X_test)):.1%}"
)

# 5. Visualisasi Flowchart Decision Tree
plt.figure(figsize=(18, 9))
plot_tree(
    model,
    feature_names=["Jam (WIB)", "Suhu (°C)", "Kelembapan (%)", "Hujan (mm)"],
    class_names=["1. Buka Jendela", "2. Tutup Jendela + AC", "3. Ngafe"],
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title(
    "Decision Tree Anak Kos Surabaya: Buka Jendela vs. AC vs. Ngafe",
    fontsize=15,
    fontweight="bold",
    pad=15,
)
plt.tight_layout()
plt.show()

from datetime import datetime


def prediksi_keputusan(
    suhu_c: float,
    hujan_mm: float,
    jam: int = None,
    kelembapan_persen: float = None,
):
  """
    Predicts the room decision based on manual temperature and precipitation inputs.
    Uses current hour and dataset median humidity as defaults if not provided.
    """
  # Default to current hour (WIB) if not manually passed
  if jam is None:
    jam = datetime.now().hour

  # Default to dataset median humidity if not manually passed
  if kelembapan_persen is None:
    kelembapan_persen = df["kelembapan_persen"].median()

  # Format input as a DataFrame with matching feature names to avoid sklearn warnings
  input_data = pd.DataFrame([{
      "jam": jam,
      "suhu_c": suhu_c,
      "kelembapan_persen": kelembapan_persen,
      "hujan_mm": hujan_mm,
  }])

  prediksi = model.predict(input_data)[0]
  return prediksi


# --- Contoh Penggunaan / Manual Input ---
if __name__ == "__main__":
  try:
    print("\n--- Cek Keputusan Kamar Kos ---")
    input_suhu = float(input("Masukkan suhu saat ini (°C): "))
    input_hujan = float(
        input("Masukkan curah hujan (mm, ketik 0 jika tidak hujan): ")
    )

    # Optional: you can also pass jam=14 explicitly to test the 'Ngafe' condition
    hasil = prediksi_keputusan(suhu_c=input_suhu, hujan_mm=input_hujan)
    print(f"\nKeputusan: {hasil}")

  except ValueError:
    print("Input tidak valid. Harap masukkan angka.")
    
    
    
    #--
    
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(15, 6))

# 1. Axis Utama (Kiri): Suhu (°C)
color_suhu = "tab:red"
ax1.set_xlabel("Waktu (WIB)")
ax1.set_ylabel("Suhu (°C)", color=color_suhu, fontweight="bold")
line1 = ax1.plot(
    df["waktu"],
    df["suhu_c"],
    color=color_suhu,
    linewidth=1.8,
    label="Suhu (°C)",
)
ax1.tick_params(axis="y", labelcolor=color_suhu)
ax1.grid(True, linestyle="--", alpha=0.4)

# 2. Axis Kedua (Kanan): Kelembapan (%)
ax2 = ax1.twinx()
color_lembap = "tab:blue"
ax2.set_ylabel("Kelembapan (%)", color=color_lembap, fontweight="bold")
line2 = ax2.plot(
    df["waktu"],
    df["kelembapan_persen"],
    color=color_lembap,
    linewidth=1.5,
    alpha=0.8,
    label="Kelembapan (%)",
)
ax2.tick_params(axis="y", labelcolor=color_lembap)

# 3. Axis Ketiga (Kanan Digeser): Curah Hujan (mm)
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 60))  # Geser axis ke kanan agar tidak bertumpuk
color_hujan = "tab:cyan"
ax3.set_ylabel("Curah Hujan (mm)", color="teal", fontweight="bold")
bars = ax3.bar(
    df["waktu"],
    df["hujan_mm"],
    color=color_hujan,
    width=0.04,
    alpha=0.6,
    label="Hujan (mm)",
)
ax3.tick_params(axis="y", labelcolor="teal")
ax3.set_ylim(0, max(df["hujan_mm"].max() * 1.2, 5))  # Jaga skala tetap rapi saat hujan 0 mm

# Format tanggal di sumbu X
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))

# Gabungkan semua legenda menjadi satu kotak
lines = line1 + line2 + [bars]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper left", frameon=True)

plt.title(
    "Data Cuaca Surabaya per Jam: Suhu, Kelembapan, dan Curah Hujan",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
fig.tight_layout()
plt.show()
    