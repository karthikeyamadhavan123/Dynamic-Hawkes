# scripts/download_chicago_data.py
import pandas as pd
import requests
import io
import os

def download_chicago_crime_data():
    print("Downloading Chicago crime data...")

    url = "https://data.cityofchicago.org/resource/ijzp-q8t2.csv"
    params = {
        '$where': "date between '2020-03-01T00:00:00' and '2020-12-19T23:59:59'",
        '$limit': 50000
    }

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()

        df = pd.read_csv(io.StringIO(response.text))
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv('data/raw/chicago_crime.csv', index=False)

        print(f"✅ Saved {len(df)} Chicago crime records")
        return df

    except Exception as e:
        print(f"❌ Error downloading: {e}")
        return None


def process_chicago_data():
    try:
        df = pd.read_csv('data/raw/chicago_crime.csv')

        df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['timestamp'])

        # normalize timestamp properly
        min_t = df['timestamp'].min().value
        max_t = df['timestamp'].max().value
        df['normalized_time'] = (df['timestamp'].astype('int64') - min_t) / (max_t - min_t)

        processed_data = []
        for _, row in df.iterrows():
            if pd.notna(row.get('community_area')):
                processed_data.append({
                    'timestamp': row['normalized_time'],
                    'community_area': int(row['community_area']),
                    'primary_type': row['primary_type']
                })

        processed_df = pd.DataFrame(processed_data)
        os.makedirs("data/processed", exist_ok=True)
        processed_df.to_csv('data/processed/chicago_crime_processed.csv', index=False)

        print(f"✅ Processed {len(processed_df)} crime events")
        return processed_df

    except Exception as e:
        print(f"❌ Error processing Chicago data: {e}")
        return None


if __name__ == "__main__":
    df = download_chicago_crime_data()
    if df is not None and not df.empty:
        process_chicago_data()
