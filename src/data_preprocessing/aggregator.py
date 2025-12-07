import pandas as pd
import os

# -------------------------
# 1. File paths
# -------------------------
twitter_file = "outputs/trump_all_weeks_combined.csv"   # Twitter combined
reddit_file = "reddit/reddit_combined_22k.csv"          # Reddit combined
output_file = "combined_text_dataset.csv"

# -------------------------
# 2. Load datasets
# -------------------------
print("📥 Loading Twitter CSV...")
df_twitter = pd.read_csv(twitter_file, lineterminator="\n", usecols=["Text"])
df_twitter.rename(columns={"Text": "text"}, inplace=True)

print("📥 Loading Reddit CSV...")
df_reddit = pd.read_csv(reddit_file, lineterminator="\n", usecols=["body"])
df_reddit.rename(columns={"body": "text"}, inplace=True)

# -------------------------
# 3. Combine
# -------------------------
print("🔗 Combining datasets...")
df_combined = pd.concat([df_twitter, df_reddit], ignore_index=True)

# Drop empty rows if any
df_combined.dropna(subset=["text"], inplace=True)

# -------------------------
# 4. Save final CSV
# -------------------------
df_combined.to_csv(output_file, index=False)

print(f"✅ Combined dataset saved: {output_file}")
print(f"📊 Total rows: {len(df_combined)}")
