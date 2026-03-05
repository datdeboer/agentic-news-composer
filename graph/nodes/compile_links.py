"""LangGraph node: compile 5 trending/popular links."""
import json
import logging
import os

from langchain_openai import ChatOpenAI

from graph.state import NewsComposerState

logger = logging.getLogger(__name__)


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    )


def compile_links(state: NewsComposerState) -> dict:
    """Select 5 trending links worth sharing, with a brief reason for each."""
    print("[compile_links] Compiling 5 trending links...")

    articles = state["raw_articles"]
    if not articles:
        print("[compile_links] No articles available.")
        return {"top_5_links": []}

    # Prepare list with social scores
    article_list = ""
    for i, a in enumerate(articles, 1):
        score = a.get("score", 0)
        article_list += f"{i}. [{a['source']} | score:{score}] {a['title']}\n   {a['url']}\n"

    prompt = f"""You are a tech news curator. From the articles below, pick the 5 most interesting,
trending, or share-worthy links. Prefer articles with high social scores and broad appeal.

Return ONLY a JSON array of exactly 5 objects with keys: "title", "url", "reason".
"reason" should be one sentence explaining why this link is worth reading. No other text.

Articles:
{article_list}"""

    llm = _get_llm()
    response = llm.invoke(prompt)
    content = response.content.strip()

    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        links = json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse links: %s", e)
        links = [
            {"title": a["title"], "url": a["url"], "reason": "High relevance to current topics."}
            for a in articles[:5]
        ]

    print(f"[compile_links] Done: {len(links)} links compiled.")
    return {"top_5_links": links[:5]}
