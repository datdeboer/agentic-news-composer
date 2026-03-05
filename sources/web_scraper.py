"""Web scraper for sites without RSS feeds using requests + BeautifulSoup."""
import asyncio
import logging

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Extra sites to scrape directly (no RSS)
SCRAPE_TARGETS = [
    {"url": "https://news.ycombinator.com", "name": "Hacker News Front Page"},
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; NewsComposerBot/1.0)"
    )
}


def _scrape_url(target: dict) -> list[dict]:
    """Scrape article titles and links from a URL."""
    url = target["url"]
    name = target.get("name", url)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        articles = []
        # Generic: collect all anchor tags that look like article links
        seen = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True)
            if not text or len(text) < 15:
                continue
            if href in seen:
                continue
            seen.add(href)
            # Resolve relative URLs
            if href.startswith("/"):
                from urllib.parse import urlparse
                parsed = urlparse(url)
                href = f"{parsed.scheme}://{parsed.netloc}{href}"
            if not href.startswith("http"):
                continue
            articles.append({
                "title": text[:200],
                "url": href,
                "content": "",
                "source": name,
                "score": 0,
            })
            if len(articles) >= 20:
                break
        logger.info("Scraper %s: %d links", name, len(articles))
        return articles
    except Exception as e:
        logger.warning("Scraper %s failed: %s", name, e)
        return []


async def fetch_scraped_articles() -> list[dict]:
    """Scrape all configured targets concurrently."""
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(None, _scrape_url, target)
        for target in SCRAPE_TARGETS
    ]
    results = await asyncio.gather(*tasks)
    articles = [a for batch in results for a in batch]
    logger.info("Scraper total: %d articles", len(articles))
    return articles
