# Prompt Instructions

All LLM prompts are in `graph/nodes/`. No app restart is needed after editing — prompts are read fresh on every run.

---

## Article Summaries

**File:** `graph/nodes/summarize.py` — `prompt` variable (~line 38)

Controls how each of the top 5 articles is summarized. Default: 2-3 sentence neutral summary.

Example changes:
- Make summaries longer: `"Summarize each article in 4-5 sentences"`
- Change tone: `"You are a cynical tech journalist. Summarize each article with a critical eye."`
- Add structure: `"Summarize each article with: one sentence on what happened, one on why it matters."`

---

## Trending Links Selection

**File:** `graph/nodes/compile_links.py` — `prompt` variable (~line 35)

Controls which 5 links are selected and the one-sentence reason shown next to each.

Example changes:
- Bias toward controversy: `"Prefer articles that challenge mainstream narratives."`
- Audience focus: `"Pick links most useful to software engineers, not business readers."`

---

## Article Ranking

**File:** `graph/nodes/rank_and_filter.py` — `prompt` variable (~line 37)

Controls how articles are scored 0–10 for relevance to your configured topics. Higher scores = more likely to appear in summaries and links.

Example changes:
- Stricter relevance: `"Only score above 7 if the article directly discusses one of the topics. Score everything else 0-3."`
- Recency bias: `"Prefer breaking news; penalise opinion pieces and evergreen content."`

---

## Blog Draft Styles

**File:** `graph/nodes/draft_blog_posts.py` — `STYLE_PROMPTS` dict (lines 15-30)

Three keys, one prompt each:

### `"opinion"`
Opinion/editorial piece. Default: 600-900 words, first person, clear stance.

### `"newsletter"`
Newsletter recap. Default: 500-700 words, warm conversational tone.

### `"deep-dive"`
Analytical deep-dive. Default: 800-1200 words, expert perspective.

Example changes to any style:
- Change length: adjust the word count in the prompt
- Change audience: add `"Written for a non-technical audience."` or `"Assume the reader is a senior engineer."`
- Change structure: replace the `Structure:` line with your own outline

---

## Revision Prompt (Request Changes)

**File:** `graph/nodes/regenerate_drafts.py` — `prompt` variable (~line 35)

Used when you request changes on a draft during review. The reviewer's notes are injected as `{notes}`. You can change how the model interprets those notes, e.g. making it more conservative or more creative in its revisions.
