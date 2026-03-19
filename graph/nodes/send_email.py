"""LangGraph node: send finalized digest as newsletter via Brevo API."""
import os
from datetime import date

import markdown
import requests

from graph.state import NewsComposerState

BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"


def send_email(state: NewsComposerState) -> dict:
    """Send the digest as an HTML email. Skipped silently if credentials are not configured."""
    api_key = os.environ.get("BREVO_API_KEY")
    if not api_key:
        print("[send_email] BREVO_API_KEY not set — skipping.", flush=True)
        return {"email_sent": False}

    from_email = os.environ.get("BREVO_FROM_EMAIL")
    from_name = os.environ.get("BREVO_FROM_NAME", "Agentic News Composer")
    to_emails = [
        addr.strip()
        for addr in os.environ.get("BREVO_TO_EMAILS", "").split(",")
        if addr.strip()
    ]

    if not from_email:
        print("[send_email] BREVO_FROM_EMAIL not set — skipping.", flush=True)
        return {"email_sent": False}

    if not to_emails:
        print("[send_email] BREVO_TO_EMAILS not set — skipping.", flush=True)
        return {"email_sent": False}

    subject, body_md = _build_content(state)
    body_html = _md_to_html(body_md)

    payload = {
        "sender": {"name": from_name, "email": from_email},
        "to": [{"email": addr} for addr in to_emails],
        "subject": subject,
        "htmlContent": body_html,
        "textContent": body_md,
    }

    response = requests.post(
        BREVO_API_URL,
        json=payload,
        headers={
            "api-key": api_key,
            "accept": "application/json",
            "content-type": "application/json",
        },
        timeout=30,
    )
    response.raise_for_status()

    message_id = response.json().get("messageId", "")
    print(f"[send_email] Digest sent to: {', '.join(to_emails)} (messageId: {message_id})", flush=True)
    return {"email_sent": True}


def _build_content(state: NewsComposerState) -> tuple[str, str]:
    """Return (subject, markdown body) built from graph state."""
    today = date.today().strftime("%B %-d, %Y")
    subject = f"News Digest — {today}"

    lines = [f"# News Digest — {today}\n"]

    summaries = state.get("top_5_summaries", [])
    if summaries:
        lines.append("## Top 5 Stories\n")
        for i, s in enumerate(summaries, 1):
            lines.append(f"### {i}. [{s.get('title', '')}]({s.get('url', '')})\n")
            lines.append(f"{s.get('summary', '')}\n")

    links = state.get("top_5_links", [])
    if links:
        lines.append("## Trending Links\n")
        for i, l in enumerate(links, 1):
            lines.append(
                f"{i}. **[{l.get('title', '')}]({l.get('url', '')})**  — {l.get('reason', '')}\n"
            )

    drafts = state.get("blog_drafts", [])
    for draft in drafts:
        style = draft.get("style", "unknown")
        lines.append(f"\n---\n\n## Blog Draft: {style.title()}\n")
        lines.append(draft.get("content", "") + "\n")

    return subject, "\n".join(lines)


def _md_to_html(md_text: str) -> str:
    body = markdown.markdown(md_text, extensions=["extra"])
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: Georgia, serif; max-width: 680px; margin: 40px auto; color: #222; line-height: 1.6; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
  h2 {{ margin-top: 2em; color: #444; }}
  h3 {{ color: #555; }}
  a {{ color: #1a73e8; }}
  hr {{ border: none; border-top: 1px solid #ddd; margin: 2em 0; }}
</style>
</head>
<body>
{body}
</body>
</html>"""
