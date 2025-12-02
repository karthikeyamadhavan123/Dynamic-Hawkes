# scripts/download_reddit_data.py
import praw
import pandas as pd
import re

def download_reddit_data():
    """Download Reddit hyperlink propagation data"""
    print("Downloading Reddit data via PRAW API...")

    reddit = praw.Reddit(
        client_id="g1ZL6K1xPXThBNVimDDw4g",
        client_secret="Nqi3LMg9lxMc2CmjzHOk0dmcynZotg",
        user_agent="personal use script by /u/veerendra22"
    )

    subreddits = [
        'news', 'worldnews', 'politics', 'science', 'technology', 
        'programming', 'machinelearning', 'datascience', 'python',
        'javascript', 'webdev', 'gaming', 'movies', 'music', 'books',
        'sports', 'fitness', 'food', 'travel', 'art', 'photography',
        'history', 'space', 'askscience', 'explainlikeimfive'
    ]

    events = []

    for subreddit_name in subreddits:
        print(f"📡 Fetching from r/{subreddit_name}...")

        try:
            subreddit = reddit.subreddit(subreddit_name)

            # Posts
            for post in subreddit.hot(limit=100):
                if post.selftext:
                    linked_subs = re.findall(r'/r/(\w+)', post.selftext)
                    for linked in linked_subs:
                        if linked.lower() in [s.lower() for s in subreddits]:
                            events.append({
                                'timestamp': post.created_utc,
                                'source': subreddit_name,
                                'target': linked.lower()
                            })

            # Comments
            for comment in subreddit.comments(limit=200):
                if hasattr(comment, 'body'):
                    linked_subs = re.findall(r'/r/(\w+)', comment.body)
                    for linked in linked_subs:
                        if linked.lower() in [s.lower() for s in subreddits]:
                            events.append({
                                'timestamp': comment.created_utc,
                                'source': subreddit_name,
                                'target': linked.lower()
                            })

        except Exception as e:
            print(f"⚠️ Error fetching r/{subreddit_name}: {e}")
            continue

    df = pd.DataFrame(events)

    if df.empty:
        print("❌ No events collected. Try running again later (rate limits).")
        return None

    # ✅ Normalize timestamps (0 to 1)
    min_t = df['timestamp'].min()
    max_t = df['timestamp'].max()
    df['normalized_time'] = (df['timestamp'] - min_t) / (max_t - min_t)

    df.to_csv('data/raw/reddit_hyperlinks.csv', index=False)
    print(f"✅ Saved {len(df)} Reddit hyperlink events → data/raw/reddit_hyperlinks.csv")

    return df

if __name__ == "__main__":
    download_reddit_data()
