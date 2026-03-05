"""LangGraph node: use LLM to rank articles by topic relevance."""
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


def rank_and_filter(state: NewsComposerState) -> dict:
    """Rank raw articles by relevance to the configured topics using an LLM."""
    print("[rank_and_filter] Ranking articles by topic relevance...")

    articles = state["raw_articles"]
    topics = state["topics"]

    if not articles:
        print("[rank_and_filter] No articles to rank.")
        return {"raw_articles": []}

    # Prepare a compact article list for the LLM (title + source, truncated content)
    article_summaries = []
    for i, a in enumerate(articles):
        content_snippet = (a.get("content") or "")[:300]
        article_summaries.append(
            f"{i}: [{a['source']}] {a['title']}\n   {content_snippet}"
        )

    # Process in batches of 50 to avoid token limits
    batch_size = 50
    scored_indices = []

    for batch_start in range(0, len(article_summaries), batch_size):
        batch = article_summaries[batch_start: batch_start + batch_size]
        batch_text = "\n\n".join(batch)

        prompt = f"""You are a news curator. Given these topics of interest:
{', '.join(topics)}

Rate each of the following articles by relevance to these topics on a scale of 0-10.
Return ONLY a JSON array of objects with keys "index" (original article index) and "score" (0-10 integer).
Do not include any other text.

Articles:
{batch_text}"""

        llm = _get_llm()
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        try:
            ratings = json.loads(content)
            for item in ratings:
                idx = batch_start + (item["index"] - batch_start if batch_start > 0 else item["index"])
                # Handle both absolute and relative indices
                raw_idx = item["index"]
                scored_indices.append({"index": raw_idx, "score": item["score"]})
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse LLM ranking response: %s\nContent: %s", e, content[:500])
            # Fallback: assign score 0 to all articles in this batch
            for i in range(batch_start, min(batch_start + batch_size, len(articles))):
                scored_indices.append({"index": i, "score": articles[i].get("score", 0) // 100})

    # Apply scores back to articles
    for item in scored_indices:
        idx = item["index"]
        if 0 <= idx < len(articles):
            articles[idx]["score"] = item["score"]

    # Sort by score descending, keep top 20 for downstream nodes
    ranked = sorted(articles, key=lambda a: a.get("score", 0), reverse=True)
    top_articles = ranked[:20]

    _save_articles(top_articles)
    print(f"[rank_and_filter] Done: top {len(top_articles)} articles selected.")
    return {"raw_articles": top_articles}


def _save_articles(articles: list[dict]) -> None:
    from output.paths import dated_path
    path = dated_path("articles")
    lines = ["# Ranked Articles\n"]
    for i, a in enumerate(articles, 1):
        lines.append(f"## {i}. {a.get('title', 'Untitled')} (score: {a.get('score', 0)})")
        lines.append(f"**Source:** {a.get('source', '')}  ")
        lines.append(f"**URL:** {a.get('url', '')}  ")
        content = a.get("content", "").strip()
        if content:
            lines.append(f"\n{content[:500]}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[rank_and_filter] Articles saved to {path}", flush=True)
