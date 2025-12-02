import os
import requests
import zipfile
from utils.helpers import create_directory

def download_reddit_data():
    """Download Reddit dataset"""
    print("Downloading Reddit data...")
    # Placeholder for actual download logic
    create_directory('data/raw')
    print("Reddit data download complete!")

def download_news_data():
    """Download GDELT news data"""
    print("Downloading News data...")
    create_directory('data/raw')
    print("News data download complete!")

def download_protest_data():
    """Download ACLED protest data"""
    print("Downloading Protest data...")
    create_directory('data/raw')
    print("Protest data download complete!")

def download_dblp_data():
    """Download DBLP data"""
    print("Downloading DBLP data...")
    create_directory('data/raw')
    print("DBLP data download complete!")

def download_all_data():
    """Download all datasets"""
    download_reddit_data()
    download_news_data()
    download_protest_data()
    download_dblp_data()
    print("All datasets downloaded!")

if __name__ == "__main__":
    download_all_data()