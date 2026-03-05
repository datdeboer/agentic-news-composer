"""Social signal fetcher — HN Algolia API + Reddit JSON API (no keys needed)."""
import asyncio
import logging
from pathlib import Path

import requests
import yaml

logger = logging.getLogger(__name__)

TOPICS_CONFIG = Path(__file__).parent.parent / "config" / "topics.yaml"

HEADERS = {"User-Agent": "NewsComposerBot/1.0 (learning project)"}


def _load_config() -> dict:
    with open(TOPICS_CONFIG) as f:
        return yaml.safe_load(f)


def _fetch_hn(top_n: int = 30) -> list[dict]:
    """Fetch top HN stories using Algolia API."""
    print(f"  [HN] Fetching top {top_n} stories...")
    try:
        url = f"https://hn.algolia.com/api/v1/search?tags=front_page&hitsPerPage={top_n}"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        articles = []
        for hit in data.get("hits", []):
            story_url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
            articles.append({
                "title": hit.get("title", ""),
                "url": story_url,
                "content": "",
                "source": "Hacker News",
                "score": hit.get("points", 0),
            })
        print(f"  [HN] Done — {len(articles)} stories")
        return articles
    except Exception as e:
        print(f"  [HN] FAILED — {e}")
        return []


def _fetch_reddit_subreddit(subreddit: str, top_n: int = 10) -> list[dict]:
    """Fetch top posts from a subreddit using Reddit's JSON API."""
    print(f"  [Reddit] Fetching r/{subreddit}...")
    try:
        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={top_n}"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        articles = []
        for post in data.get("data", {}).get("children", []):
            p = post["data"]
            if p.get("stickied"):
                continue
            articles.append({
                "title": p.get("title", ""),
                "url": p.get("url", ""),
                "content": p.get("selftext", "")[:500],
                "source": f"r/{subreddit}",
                "score": p.get("score", 0),
            })
        print(f"  [Reddit] r/{subreddit}: {len(articles)} posts")
        return articles
    except Exception as e:
        print(f"  [Reddit] r/{subreddit}: FAILED — {e}")
        return []


async def fetch_social_articles() -> list[dict]:
    """Fetch HN + Reddit stories concurrently."""
    cfg = _load_config()
    subreddits = cfg.get("reddit_subreddits", ["technology"])
    hn_top_n = cfg.get("hn_top_n", 30)
    reddit_top_n = cfg.get("reddit_top_n", 10)

    print(f"[Social] Fetching HN + {len(subreddits)} subreddits in parallel...")
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, _fetch_hn, hn_top_n)]
    for sub in subreddits:
        tasks.append(loop.run_in_executor(None, _fetch_reddit_subreddit, sub, reddit_top_n))

    results = await asyncio.gather(*tasks)
    articles = [a for batch in results for a in batch]
    print(f"[Social] Done — {len(articles)} total articles")
    return articles
