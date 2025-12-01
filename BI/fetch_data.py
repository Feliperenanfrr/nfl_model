import os
import requests
import pandas as pd

DATA_DIR = "BI/data"
os.makedirs(DATA_DIR, exist_ok=True)

def download_file(url, filename):
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"{filename} already exists. Skipping.")
        return

    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}.")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

def main():
    # 1. Download Draft Picks
    draft_url = "https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.csv"
    download_file(draft_url, "draft_picks.csv")

    # 2. Download Player Stats (Weekly) for 2013-2023
    # We will aggregate these into a single file later or load them as needed.
    base_stats_url = "https://github.com/nflverse/nflverse-data/releases/download/player_stats/stats_player_week_{year}.csv"
    
    for year in range(2013, 2024):
        url = base_stats_url.format(year=year)
        filename = f"player_stats_{year}.csv"
        download_file(url, filename)

if __name__ == "__main__":
    main()
