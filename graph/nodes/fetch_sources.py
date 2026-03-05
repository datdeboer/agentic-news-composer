"""LangGraph node: fetch articles from all sources in parallel."""
import asyncio
import logging
from pathlib import Path

import yaml

from graph.state import NewsComposerState
from sources.rss_fetcher import fetch_rss_articles
from sources.social_fetcher import fetch_social_articles
from sources.web_scraper import fetch_scraped_articles

logger = logging.getLogger(__name__)

TOPICS_CONFIG = Path(__file__).parent.parent.parent / "config" / "topics.yaml"


def _load_topics() -> list[str]:
    with open(TOPICS_CONFIG) as f:
        return yaml.safe_load(f).get("topics", [])


def fetch_sources(state: NewsComposerState) -> dict:
    """Fetch articles from RSS feeds, web scraping, and social APIs concurrently."""
    print("[fetch_sources] Starting parallel fetch from all sources...")

    topics = state.get("topics") or _load_topics()

    async def _gather():
        rss, scraped, social = await asyncio.gather(
            fetch_rss_articles(),
            fetch_scraped_articles(),
            fetch_social_articles(),
        )
        return rss + scraped + social

    all_articles = asyncio.run(_gather())

    # Deduplicate by URL
    seen_urls = set()
    unique_articles = []
    for article in all_articles:
        url = article.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_articles.append(article)

    print(f"[fetch_sources] Done: {len(unique_articles)} unique articles fetched.")
    return {
        "topics": topics,
        "raw_articles": unique_articles,
    }
