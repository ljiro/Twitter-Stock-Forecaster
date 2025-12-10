from Scweet.scweet import Scweet 
import os
import glob
import shutil
import time
from dotenv import load_dotenv

# -------------------------
# 1. Configuration
# -------------------------
TEMP_DIR = "data/temp_buffer"
OUTPUT_DIR = "src/data_collection/outputs"
COOKIES_DIR = "src/data_collection/cookies"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(COOKIES_DIR, exist_ok=True)

# -------------------------
# 2. Helpers
# -------------------------
def setup_env():
    # Load from project root .env
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    env_path = os.path.join(root_dir, ".env")
    
    load_dotenv(env_path)
    
    email = os.getenv('SCWEET_EMAIL')
    username = os.getenv('SCWEET_USERNAME')
    password = os.getenv('SCWEET_PASSWORD')
    
    if email: os.environ["EMAIL"] = email
    if username: os.environ["USERNAME"] = username
    if password: os.environ["PASSWORD"] = password

def wait_for_file_release(filepath, timeout=10):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            os.rename(filepath, filepath)
            return True
        except OSError:
            time.sleep(1)
    return False

# -------------------------
# 3. Scraper Function
# -------------------------
def run_scraper(start_date=None, end_date=None, words=None):
    if words is None:
        words = ['NVIDIA OR Nvidia OR nvidia OR $NVDA OR NVDA']
    
    if not start_date or not end_date:
        return None

    setup_env()

    try:
        print(f"\n🚀 Starting Scrape: {start_date} -> {end_date}")
        
        # Clean temp buffer
        for f in glob.glob(os.path.join(TEMP_DIR, "*")):
            try: os.remove(f)
            except: pass

        # Initialize Scweet
        scraper = Scweet(
            cookies_path=COOKIES_DIR,
            headless=False,
            disable_images=True
        )
        
        # --- RUN SCRAPE (With Limit) ---
        scraper.scrape(
            since=start_date,
            until=end_date,
            words=words,
            from_account=None,
            display_type="latest",
            lang="en",
            save_dir=TEMP_DIR,
            limit=50  # <--- STOP AFTER 50 TWEETS
        )
        
        # --- ROBUST FILE MOVING ---
        time.sleep(2)
        
        files = glob.glob(os.path.join(TEMP_DIR, "*.csv"))
        
        if not files:
            print("❌ File not found in temp. Scrape returned 0 results.")
            return None
            
        scweet_file = max(files, key=os.path.getmtime)
        final_filename = f"scrape_{start_date}_{end_date}.csv"
        target_path = os.path.join(OUTPUT_DIR, final_filename)
        
        if not wait_for_file_release(scweet_file):
            print("❌ Error: File is permanently locked by Chrome.")
            return None

        if os.path.exists(target_path):
            try: os.remove(target_path)
            except: pass
            
        try:
            shutil.move(scweet_file, target_path)
            print(f"✅ Scrape Success: {target_path}")
            return target_path
        except:
            shutil.copy2(scweet_file, target_path)
            print(f"✅ Scrape Success (Copied): {target_path}")
            return target_path

    except Exception as e:
        print(f"❌ Scraper Error: {e}")
        return None