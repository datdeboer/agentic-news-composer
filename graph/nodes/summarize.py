"""LangGraph node: generate 5 article summaries using LLM."""
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


def summarize(state: NewsComposerState) -> dict:
    """Generate concise summaries for the top 5 articles."""
    print("[summarize] Generating summaries for top 5 articles...")

    articles = state["raw_articles"][:5]
    if not articles:
        print("[summarize] No articles to summarize.")
        return {"top_5_summaries": []}

    article_text = ""
    for i, a in enumerate(articles, 1):
        content = (a.get("content") or "")[:500]
        article_text += f"\n{i}. Title: {a['title']}\n   URL: {a['url']}\n   Content: {content}\n"

    prompt = f"""You are a news analyst. Summarize each of the following articles in 2-3 sentences.
Return ONLY a JSON array of objects with keys: "title", "url", "summary".
Use the original title and URL from the article. Do not include any other text.

Articles:
{article_text}"""

    llm = _get_llm()
    response = llm.invoke(prompt)
    content = response.content.strip()

    # Strip markdown code fences
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        summaries = json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse summaries: %s", e)
        summaries = [
            {"title": a["title"], "url": a["url"], "summary": a.get("content", "")[:200]}
            for a in articles
        ]

    print(f"[summarize] Done: {len(summaries)} summaries generated.")
    return {"top_5_summaries": summaries[:5]}
