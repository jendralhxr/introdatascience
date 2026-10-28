import math
import re
import time
from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests as raw_requests
from sklearn.linear_model import LinearRegression

# Base search URL template
BASE_URL = "https://www.mudah.my/malaysia/all?q=civic&o={page}"


def fetch_page_soup(page_num: int) -> BeautifulSoup:
    """Fetches page HTML using curl_cffi (Chrome TLS fingerprint impersonation)

    with standard requests fallback.
    """
    url = BASE_URL.format(page=page_num)

    # 1. Attempt fetching via curl_cffi (Bypasses Cloudflare TLS protection)
    try:
        res = curl_requests.get(
            url, impersonate="chrome120", timeout=15, follow_redirects=True
        )
        if res.status_code == 200:
            return BeautifulSoup(res.content, "html.parser")
    except Exception as e:
        print(f"curl_cffi fetch failed on page {page_num}: {e}")

    # 2. Fallback to standard requests if curl_cffi fails
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    res_raw = raw_requests.get(url, headers=headers, timeout=15)
    return BeautifulSoup(res_raw.content, "html.parser")


def parse_page_listings(soup: BeautifulSoup, page_num: int) -> list:
    """Parses text blocks from rendered body content into structured records."""
    raw_text = soup.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

    records = []

    for i, line in enumerate(lines):
        # Match mileage range patterns (e.g., '100000 - 109999' or '0 - 4999')
        m_mileage = re.match(r"^(\d+)\s*-\s*(\d+)$", line)
        if m_mileage:
            low, high = int(m_mileage.group(1)), int(m_mileage.group(2))
            avg_mileage = (low + high) / 2.0

            # Next line after mileage is Year (e.g., 2018)
            year = (
                int(lines[i + 1])
                if (
                    i + 1 < len(lines)
                    and re.match(r"^(19|20)\d{2}$", lines[i + 1])
                )
                else None
            )

            # Inspect preceding 6 lines for Price and Title
            back_lines = lines[max(0, i - 6) : i]
            prices = []
            for bline in back_lines:
                pm = re.search(r"RM\s*([\d\s,]+)", bline)
                if pm:
                    clean_p = pm.group(1).replace(" ", "").replace(",", "")
                    if clean_p.isdigit():
                        prices.append(float(clean_p))

            price = prices[-1] if prices else None

            # Extract Title containing 'civic'
            title = None
            for bline in back_lines:
                if "civic" in bline.lower():
                    title = bline
                    break
            if not title and back_lines:
                title = back_lines[0]

            if price and year and title and "civic" in title.lower():
                records.append({
                    "page": page_num,
                    "title": title,
                    "price_rm": price,
                    "mileage_km": avg_mileage,
                    "year": year,
                })

    return records


# ==========================================
# 1. Multi-Page Scraping Loop
# ==========================================
max_pages = 20
scraped_data = []

print("Starting scraper with curl_cffi & requests fallback...")

for p in range(1, max_pages + 1):
    print(f"Fetching Page {p}...")
    soup = fetch_page_soup(p)
    page_listings = parse_page_listings(soup, p)
    print(f"  -> Captured {len(page_listings)} valid Civic listings.")
    scraped_data.extend(page_listings)
    time.sleep(2)  # Respectful delay between requests

df = pd.DataFrame(scraped_data)

if df.empty:
    raise RuntimeError("No data extracted. Verify network access and website response.")

# Exclude high-value trim outliers (e.g., Civic Type R > RM 190,000) for standard regression modeling
df_std = df[df["price_rm"] < 190000].copy()

# ==========================================
# 2. Linear Regression Modeling
# ==========================================
X = df_std[["mileage_km"]]
y = df_std["price_rm"]

model = LinearRegression()
model.fit(X, y)

slope = model.coef_[0]
intercept = model.intercept_

print("\n--- Model Results ---")
print(f"Total Listings Scraped: {len(df)}")
print(
    f"Model Equation: Price (RM) = {intercept:,.2f} + ({slope:.4f} * Mileage_KM)"
)
print(
    f"Depreciation Rate: ~RM {abs(slope * 10000):,.2f} drop per 10,000 km driven"
)

# ==========================================
# 3. Data Visualization
# ==========================================
plt.figure(figsize=(12, 7))

# Scatter plot of listings
plt.scatter(
    df["mileage_km"],
    df["price_rm"],
    color="#e74c3c",
    edgecolors="black",
    s=70,
    alpha=0.75,
    label="Scraped Listings",
)

# Regression line
x_vals = np.linspace(
    df_std["mileage_km"].min(), df_std["mileage_km"].max(), 100
).reshape(-1, 1)
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
    f"Honda Civic: Price vs. Mileage (Mudah.my Scraped Data, N={len(df)})",
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