from Scweet.scweet import scrape 
from pyvirtualdisplay import Display
import os

OUTPUT_DIR = "src/data_collection/outputs"
COOKIES_DIR = "src/data_collection/cookies"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(COOKIES_DIR, exist_ok=True)

def run_scraper(start_date=None, end_date=None, words=None):
    if words is None:
        words = ['"Pfizer" OR "Pfizer drug" OR "pfizer" OR "pfizer vaccine"']
    if not start_date or not end_date: return None

    display = Display(visible=0, size=(1920, 1080))
    display.start()
    
    try:
        print(f"\n🚀 Starting Scrape: {start_date} -> {end_date}")
        csv_filename = f"scrape_{start_date}_{end_date}.csv"
        
        # Scweet Functional Call
        scrape(
            since=start_date, until=end_date, words=words,
            from_account=None, headless=False, display_type="Latest",
            save_images=False, lang="en", resume=False, filter_replies=True,
            proximity=True, save_dir=OUTPUT_DIR, csv_name=csv_filename 
        )
        
        expected_path = os.path.join(OUTPUT_DIR, csv_filename)
        if not os.path.exists(expected_path) and os.path.exists(expected_path + ".csv"):
            expected_path += ".csv"
            
        return expected_path if os.path.exists(expected_path) else None
    except Exception as e:
        print(f"❌ Scraper Error: {e}")
        return None
    finally:
        display.stop()