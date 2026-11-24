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

# Tambahkan sedikit variasi realistis (5% noise) agar pohon keputusan belajar pola alami
np.random.seed(42)
noise_idx = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
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