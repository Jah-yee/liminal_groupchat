"""Optional web search for !search, via DuckDuckGo (pip install ddgs)."""

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None


def web_search(query, max_results=4):
    """Return search results as short plain text for the chat."""
    if DDGS is None:
        return "(search isn't installed - pip install ddgs)"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"(search failed: {e})"
    if not results:
        return "(no results)"
    lines = []
    for r in results:
        title = r.get("title", "").strip()
        body = (r.get("body") or "").strip()
        if len(body) > 220:
            body = body[:217] + "..."
        lines.append(f"- {title}: {body} ({r.get('href', '')})")
    return "\n".join(lines)
