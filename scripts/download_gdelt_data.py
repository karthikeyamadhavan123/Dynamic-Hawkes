# scripts/download_gdelt_data.py
import requests
import pandas as pd
import io
import os
import csv
import json

def try_read_csv_text(text):
    """Try reading CSV text into DataFrame with a few fallbacks."""
    # Primary: normal CSV
    try:
        return pd.read_csv(io.StringIO(text))
    except Exception:
        pass

    # Try reading as CSV with different delimiters (tab)
    try:
        return pd.read_csv(io.StringIO(text), sep='\t', engine='python')
    except Exception:
        pass

    # Try reading as JSON
    try:
        return pd.read_json(io.StringIO(text))
    except Exception:
        pass

    # Last resort: try to parse with csv.reader and build DataFrame
    try:
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            return None
        header = rows[0]
        data = rows[1:]
        return pd.DataFrame(data, columns=header)
    except Exception:
        return None

def find_url_column(df):
    """Detect a column that likely contains URLs.
       Strategy:
         1) look for common names (case-insensitive)
         2) look for any column that contains 'http' strings
    """
    if df is None or df.shape[0] == 0:
        return None

    # Candidate names to check (lowercased)
    name_candidates = [
        "documentidentifier", "sourceurl", "source_url", "url", "sourceurl",
        "uri", "link", "linkurl", "source", "sourcecommonname"
    ]
    cols_lower = {c.lower(): c for c in df.columns}

    # 1) direct name match (case-insensitive)
    for cand in name_candidates:
        if cand in cols_lower:
            return cols_lower[cand]

    # 2) substring 'url' / 'link' / 'document' heuristics
    for col in df.columns:
        low = col.lower()
        if "url" in low or "link" in low or "document" in low or "uri" in low:
            return col

    # 3) inspect contents for 'http'
    for col in df.columns:
        try:
            # sample some values to avoid heavy ops on huge frames
            sample = df[col].astype(str).head(200).str.lower()
            if sample.str.contains("http://|https://", regex=True).any():
                return col
        except Exception:
            continue

    return None

def extract_domain(series):
    """Extract domain from a series of URLs (string)."""
    return series.astype(str).str.extract(r'https?://([^/]+)/?')[0]

def download_gdelt_data(maxrecords=20000):
    print("Fetching GDELT COVID-related news (Jan 20 to Mar 24, 2020)...")
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": "covid OR coronavirus OR pandemic",
        "mode": "artlist",
        "format": "csv",
        "startdatetime": "20200120000000",
        "enddatetime": "20200324000000",
        "maxrecords": maxrecords
    }

    try:
        response = requests.get(url, params=params, timeout=60)
    except Exception as e:
        print("❌ Request error:", e)
        return None

    if response.status_code != 200:
        print("❌ Request failed:", response.status_code, response.text[:200])
        return None

    text = response.text

    # Try to parse the response robustly
    df = try_read_csv_text(text)
    if df is None:
        # save raw response for inspection and return
        os.makedirs("data/raw", exist_ok=True)
        raw_path = "data/raw/gdelt_covid_raw_response.txt"
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"❌ Could not parse response into a DataFrame. Raw response saved to {raw_path}")
        return None

    # Find a URL column (common names and content heuristic)
    url_col = find_url_column(df)

    if url_col is None:
        # No URL column found. Save the full DataFrame and warn.
        os.makedirs("data/raw", exist_ok=True)
        out_path = "data/raw/gdelt_covid_news.csv"
        df.to_csv(out_path, index=False)
        print("⚠️ No URL column found. Saved raw GDELT table to", out_path)
        print("Columns present:", list(df.columns))
        print("Tip: inspect the raw response (data/raw/gdelt_covid_raw_response.txt) to locate the URL field or adjust the script.")
        return df

    # Extract domain
    try:
        df['source_domain'] = extract_domain(df[url_col])
    except Exception:
        df['source_domain'] = None

    # Keep a small set of useful columns if available
    keep_cols = []
    for candidate in ["Date", "SourceCommonName", url_col, "source_domain"]:
        if candidate in df.columns:
            keep_cols.append(candidate)

    # Always include the URL column and the source_domain we've computed
    if url_col not in keep_cols:
        keep_cols.append(url_col)
    if "source_domain" not in keep_cols:
        keep_cols.append("source_domain")

    df_out = df.loc[:, keep_cols]

    os.makedirs("data/raw", exist_ok=True)
    out_path = "data/raw/gdelt_covid_news.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Saved dataset: {out_path}")
    print(f"Total records: {len(df_out)}")
    print(f"Detected URL column: {url_col}")
    return df_out

if __name__ == "__main__":
    download_gdelt_data()
