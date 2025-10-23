
from dotenv import load_dotenv                            
import os
from Scweet.scweet import Scweet
import pandas as pd
import sys
from datetime import datetime, timedelta

# -------------------------
# 1. Setup
# -------------------------
dotenv_path = '/path/.env'
load_dotenv(dotenv_path)
os.makedirs("outputs", exist_ok=True)
os.makedirs("cookies", exist_ok=True)

# -------------------------
# 2. Date ranges (weekly windows)
# -------------------------
def generate_weekly_ranges(start_date, end_date):
    ranges = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while current < end:
        week_end = min(current + timedelta(days=6), end)
        ranges.append((current.strftime("%Y-%m-%d"), week_end.strftime("%Y-%m-%d")))
        current = week_end + timedelta(days=1)
    return ranges

date_ranges = generate_weekly_ranges("2025-01-01", "2025-09-17")

# -------------------------
# 3. Keywords
# -------------------------
keywords = ['"Pfizer" OR "Pfizer drug" OR "pfizer" OR "pfizer vaccine"']

# -------------------------
# 4. Scrape one week only
# -------------------------
def scrape_one_week(week_index):
    start, end = date_ranges[week_index]
    scweet = Scweet(cookies_path="cookies", disable_images=True, headless=False)

    csv_name = f"pfizer_week{week_index+1}_{start}_to_{end}.csv"
    print(f"\n🚀 Scraping Week {week_index+1}: {start} → {end} ...")

    tweets = scweet.scrape(
        since=start,
        until=end,
        words=keywords,
        lang="en",
        limit=5000,
        save_dir="outputs",
        headless=True,
        custom_csv_name=csv_name
    )

    if isinstance(tweets, pd.DataFrame):
        df_week = tweets
    else:
        df_week = pd.DataFrame(tweets)

    print(f"✅ Finished Week {week_index+1}: {len(df_week)} tweets")

# -------------------------
# 5. Run
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Please provide a week index (0-based). Example: python scrape_week.py 2")
        sys.exit(1)

    week_index = int(sys.argv[1])
    scrape_one_week(week_index)
