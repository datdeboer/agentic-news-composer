"""RSS/Atom feed fetcher using feedparser."""
import asyncio
import logging
from pathlib import Path

import feedparser
import yaml

logger = logging.getLogger(__name__)

FEEDS_CONFIG = Path(__file__).parent.parent / "config" / "feeds.yaml"


def _load_feeds() -> list[dict]:
    with open(FEEDS_CONFIG) as f:
        return yaml.safe_load(f)["feeds"]


def _fetch_single_feed(feed: dict) -> list[dict]:
    """Fetch and parse one RSS/Atom feed synchronously."""
    url = feed["url"]
    name = feed.get("name", url)
    print(f"  [RSS] Fetching {name}...")
    try:
        parsed = feedparser.parse(url)
        articles = []
        for entry in parsed.entries[:10]:  # cap at 10 per feed
            content = ""
            if hasattr(entry, "summary"):
                content = entry.summary
            elif hasattr(entry, "content"):
                content = entry.content[0].value if entry.content else ""

            articles.append({
                "title": entry.get("title", ""),
                "url": entry.get("link", ""),
                "content": content[:2000],  # cap content length
                "source": name,
                "score": 0,
            })
        print(f"  [RSS] {name}: {len(articles)} articles")
        return articles
    except Exception as e:
        print(f"  [RSS] {name}: FAILED — {e}")
        return []


async def fetch_rss_articles() -> list[dict]:
    """Fetch articles from all configured RSS feeds concurrently."""
    feeds = _load_feeds()
    print(f"[RSS] Fetching {len(feeds)} feeds in parallel...")
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(None, _fetch_single_feed, feed)
        for feed in feeds
    ]
    results = await asyncio.gather(*tasks)
    articles = [article for batch in results for article in batch]
    print(f"[RSS] Done — {len(articles)} articles from {len(feeds)} feeds")
    return articles
