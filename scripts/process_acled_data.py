# scripts/process_acled_data.py
import pandas as pd
import os

def process_acled_data():
    print("Processing ACLED protest data...")

    try:
        df = pd.read_csv('data/raw/acled_data.csv')

        protest_keywords = ['Protest', 'Demonstration', 'Rally', 'Strike']
        protest_df = df[df['event_type'].astype(str).str.contains('|'.join(protest_keywords), case=False, na=False)]

        protest_df['timestamp'] = pd.to_datetime(protest_df['event_date'], errors='coerce')
        protest_df = protest_df.dropna(subset=['timestamp'])

        min_t = protest_df['timestamp'].min().value
        max_t = protest_df['timestamp'].max().value
        protest_df['normalized_time'] = (protest_df['timestamp'].astype('int64') - min_t) / (max_t - min_t)

        processed_df = protest_df[['normalized_time', 'country', 'event_type']]
        os.makedirs("data/processed", exist_ok=True)
        processed_df.to_csv('data/processed/acled_protests_processed.csv', index=False)

        print(f"✅ Processed {len(processed_df)} protest events")
        return processed_df

    except FileNotFoundError:
        print("❌ Please place ACLED CSV at: data/raw/acled_data.csv")
        return None

if __name__ == "__main__":
    process_acled_data()
