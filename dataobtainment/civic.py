"""
Scrape Honda Civic listings from Mudah.my and plot price vs mileage.

Install:  pip install requests beautifulsoup4 pandas matplotlib
Run:      python mudah_civic_scraper.py

NOTE: Not tested against the live site (my sandbox can't reach mudah.my).
Selectors are based on the page text I saw. If it returns 0 rows, the site
may be blocking plain requests or rendering with JavaScript. In that case
use the Playwright fallback described at the bottom.
"""

import re
import time

import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.mudah.my/malaysia/all"
QUERY = "civic"
MAX_PAGES = 10          # each page has ~40 results; raise carefully
DELAY_SECONDS = 2.5     # be polite, don't hammer the server
PAGE_PARAM = "o"        # Mudah's page parameter (?o=2). Verify in your browser.

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

PRICE_RE = re.compile(r"RM\s*([\d,\s]+)")
MILEAGE_RE = re.compile(r"\b(\d{1,6})\s*-\s*(\d{1,6})\b")
YEAR_RE = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")
CC_RE = re.compile(r"\b(\d{3,4})\s*cc\b", re.I)


def parse_price(text):
    """Return the LAST price in the text (so discounted price wins over struck-through one)."""
    matches = PRICE_RE.findall(text)
    if not matches:
        return None
    digits = re.sub(r"\D", "", matches[-1])
    return int(digits) if digits else None


def find_card(h2):
    """Walk up from the title until we reach the container holding the whole listing."""
    node = h2
    for _ in range(6):
        node = node.parent
        if node is None:
            break
        if node.find("img") and "RM" in node.get_text():
            return node
    return h2.parent


def parse_listing(h2):
    link = h2.find("a", href=True)
    if not link or not link["href"].endswith(".htm"):
        return None

    card = find_card(h2)
    # Icons carry their value in the alt text (e.g. alt="85000 - 89999"), so include both
    alts = " | ".join(img.get("alt", "") for img in card.find_all("img"))
    text = card.get_text(" | ", strip=True) + " | " + alts

    mileage = MILEAGE_RE.search(alts) or MILEAGE_RE.search(text)
    if not mileage:
        return None  # not a car (accessory, lorry, property...)

    lo, hi = int(mileage.group(1)), int(mileage.group(2))
    year = YEAR_RE.search(alts) or YEAR_RE.search(link["href"])
    cc = CC_RE.search(text)
    condition = next((c for c in ("Recon", "Used", "New") if c in text), None)

    return {
        "title": link.get_text(strip=True),
        "url": link["href"],
        "price_rm": parse_price(card.get_text(" ", strip=True)),
        "mileage_range": f"{lo}-{hi}",
        "mileage_mid": (lo + hi) / 2,
        "year": int(year.group(1)) if year else None,
        "engine_cc": int(cc.group(1)) if cc else None,
        "condition": condition,
    }


def scrape(max_pages=MAX_PAGES):
    rows, seen = [], set()
    session = requests.Session()
    session.headers.update(HEADERS)

    for page in range(1, max_pages + 1):
        params = {"q": QUERY}
        if page > 1:
            params[PAGE_PARAM] = page

        resp = session.get(BASE_URL, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"Page {page}: HTTP {resp.status_code}, stopping.")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        added = 0
        for h2 in soup.find_all("h2"):
            row = parse_listing(h2)
            if row and row["url"] not in seen:
                seen.add(row["url"])
                rows.append(row)
                added += 1

        print(f"Page {page}: {added} car listings")
        if added == 0:
            break
        time.sleep(DELAY_SECONDS)

    return pd.DataFrame(rows)


def plot(df):
    df = df.dropna(subset=["price_rm", "mileage_mid"])
    is_type_r = df["title"].str.contains("type r", case=False, na=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df.loc[~is_type_r, "mileage_mid"] / 1000,
               df.loc[~is_type_r, "price_rm"] / 1000,
               alpha=0.7, label="Other Civic")
    ax.scatter(df.loc[is_type_r, "mileage_mid"] / 1000,
               df.loc[is_type_r, "price_rm"] / 1000,
               alpha=0.8, color="red", label="Type R")
    ax.set_xlabel("Mileage (thousand km, range midpoint)")
    ax.set_ylabel("Price (RM thousand)")
    ax.set_title("Honda Civic on Mudah.my: Price vs Mileage")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("civic_price_vs_mileage.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    df = scrape()
    if df.empty:
        print("No listings parsed. See the note at the top of the file.")
    else:
        df.to_csv("mudah_civic.csv", index=False)
        print(f"Saved {len(df)} listings to mudah_civic.csv")
        plot(df)


# ---------------------------------------------------------------------------
# Playwright fallback (if plain requests returns no listings):
#   pip install playwright && playwright install chromium
#
#   from playwright.sync_api import sync_playwright
#   with sync_playwright() as p:
#       browser = p.chromium.launch()
#       page = browser.new_page()
#       page.goto("https://www.mudah.my/malaysia/all?q=civic")
#       page.wait_for_selector("h2")
#       html = page.content()
#   soup = BeautifulSoup(html, "html.parser")   # then reuse parse_listing()
# ---------------------------------------------------------------------------