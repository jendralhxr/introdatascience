import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Load and Parse Plaintext Dump
with open("mudah_text.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

extracted_listings = []

for i, line in enumerate(lines):
    # Match mileage pattern (e.g., "100000 - 109999" or "0 - 4999")
    m_mileage = re.match(r"^(\d+)\s*-\s*(\d+)$", line)
    if m_mileage:
        mileage_low = int(m_mileage.group(1))
        mileage_high = int(m_mileage.group(2))
        avg_mileage = (mileage_low + mileage_high) / 2.0

        # Next line is Year (e.g., 2012)
        year = (
            int(lines[i + 1])
            if i + 1 < len(lines) and re.match(r"^(19|20)\d{2}$", lines[i + 1])
            else None
        )

        # Engine capacity line (e.g., 1799cc)
        cc = (
            lines[i + 2]
            if i + 2 < len(lines) and re.match(r"^\d+cc$", lines[i + 2])
            else None
        )

        # Inspect preceding 6 lines for Price and Title
        back_lines = lines[max(0, i - 6) : i]
        prices_found = []
        for bline in back_lines:
            p_match = re.search(r"RM\s*([\d\s,]+)", bline)
            if p_match:
                clean_p = p_match.group(1).replace(" ", "").replace(",", "")
                if clean_p.isdigit():
                    prices_found.append(float(clean_p))

        # Capture the final discounted price if multiple prices exist
        price = prices_found[-1] if prices_found else None

        # Title extraction
        title = None
        for bline in back_lines:
            if "civic" in bline.lower():
                title = bline
                break
        if not title and back_lines:
            title = back_lines[0]

        if price and year and title and "civic" in title.lower():
            extracted_listings.append({
                "title": title,
                "price_rm": price,
                "mileage_km": avg_mileage,
                "year": year,
                "engine_cc": cc,
            })

df = pd.DataFrame(extracted_listings)

# Filter out high-performance outliers (e.g., Type R FK8/FL5 priced > RM 190,000) for a standard model regression
df_standard = df[df["price_rm"] < 190000].copy()

# 2. Linear Regression Modeling
X = df_standard[["mileage_km"]]
y = df_standard["price_rm"]

model = LinearRegression()
model.fit(X, y)

slope = model.coef_[0]
intercept = model.intercept_
r2_score = model.score(X, y)

print(f"Total Civic Listings Extracted: {len(df)}")
print(
    f"Linear Model Equation: Price (RM) = {intercept:,.2f} + ({slope:.4f} * Mileage_KM)"
)
print(
    f"Depreciation Rate: ~RM {abs(slope * 10000):,.2f} per 10,000 km driven"
)
print(f"Model R² Score: {r2_score:.4f}")

# 3. Plotting Results
plt.figure(figsize=(12, 7))

# Scatter plot of raw data
plt.scatter(
    df["mileage_km"],
    df["price_rm"],
    color="#e74c3c",
    edgecolors="black",
    s=70,
    alpha=0.75,
    label="Mudah.my Listings",
)

# Regression line
x_vals = np.linspace(df_standard["mileage_km"].min(), df_standard["mileage_km"].max(), 100).reshape(-1, 1)
y_vals = model.predict(x_vals)
plt.plot(
    x_vals,
    y_vals,
    color="#2c3e50",
    linewidth=2.5,
    linestyle="--",
    label=f"Depreciation Trend ({slope:.2f} RM/km)",
)

plt.title(
    f"Honda Civic: Price vs. Mileage Distribution (N={len(df)} Real Listings)",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Mileage (KM)", fontsize=12)
plt.ylabel("Price (RM)", fontsize=12)
plt.gca().yaxis.set_major_formatter("RM {x:,.0f}")
plt.gca().xaxis.set_major_formatter("{x:,.0f} km")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()