# Install needed libraries first:
# pip install google-play-scraper pandas

import os
from datetime import datetime, timedelta
import pandas as pd
from google_play_scraper import reviews, Sort

# --- Config ---
APP_ID = "in.swiggy.android"   # App ID on Google Play
DAYS = 30                      # How many past days to scrape
SAVE_DIR = "swiggy_reviews"    # Where to save CSV files
# ---------------

os.makedirs(SAVE_DIR, exist_ok=True)

end_date = datetime.now()
start_date = end_date - timedelta(days=DAYS)

# Fetch all reviews in one go
all_reviews, _ = reviews(
    APP_ID,
    lang="en",
    country="in",
    sort=Sort.NEWEST,
    count=10000
)

# Process day-by-day
for day in range(DAYS + 1):
    day_start = start_date + timedelta(days=day)
    day_end = day_start + timedelta(days=1)

    # Filter reviews by date
    day_reviews = [
        r for r in all_reviews
        if isinstance(r["at"], datetime) and day_start <= r["at"] < day_end
    ]

    if day_reviews:
        df = pd.DataFrame(day_reviews)
        filename = os.path.join(SAVE_DIR, f"{day_start.date()}.csv")
        df.to_csv(filename, index=False)
        print(f"✅ Saved {len(day_reviews)} reviews for {day_start.date()}")
    else:
        print(f"❌ No reviews for {day_start.date()}")