"""Commands the AIs can use inline, and pass detection."""

import re
from dataclasses import dataclass, field

Q = r'(?:"([^"]+)"|\'([^\']+)\')'  # "double" or 'single' quoted argument

PATTERNS = {
    "image": rf"!image\s+{Q}",
    "search": rf"!search\s+{Q}",
    "bsky": rf"!(?:bsky|bluesky)\s+{Q}",
    "remember": rf"!remember\s+{Q}",
    "forget": rf"!forget\s+{Q}",
    "whisper": rf"!whisper\s+{Q}\s+{Q}",
    # !react "😂" / !react 😂, optionally followed by "Name" to pick whose message
    # !poll "question" "option" "option" ...   (or the backrooms' !vote "q" [a, b])
    "poll": r'!poll\s+"([^"]+)"((?:[\s,;]*"[^"]*")+)',
    "poll_list": r'!(?:poll|vote)\s+"([^"]+)"\s*\[([^\]]+)\]',
    # !vote "option" or !vote 2 - in the latest open poll
    "vote": r'!vote\s+(?:"([^"]+)"|\'([^\']+)\'|(\d+)\b)',
    "react": r'!react\s+(?:"([^"]{1,16})"|\'([^\']{1,16})\'|([^\s"\']{1,16}))(?:[ \t]+' + Q + r')?',
}


@dataclass
class Command:
    action: str
    args: dict = field(default_factory=dict)


def _first(groups, *indices):
    for i in indices:
        if i < len(groups) and groups[i]:
            return groups[i].strip()
    return None


def parse(text):
    """Return (text with commands removed, [Command, ...]) in order of appearance."""
    found = []
    for action, pattern in PATTERNS.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            g = match.groups()
            if action == "whisper":
                args = {"to": _first(g, 0, 1), "text": _first(g, 2, 3)}
            elif action == "react":
                args = {"emoji": _first(g, 0, 1, 2), "to": _first(g, 3, 4)}
            elif action == "poll":
                args = {"question": g[0].strip(), "options": re.findall(r'"([^"]*)"', g[1])}
            elif action == "poll_list":
                action = "poll"
                args = {"question": g[0].strip(), "options": g[1].split(",")}
            elif action == "vote":
                args = {"choice": _first(g, 0, 1, 2)}
            else:
                args = {"text": _first(g, 0, 1)}
            found.append((match.start(), match.group(0), Command(action, args)))
    found.sort(key=lambda f: (f[0], -len(f[1])))
    # One command per stretch of text: `!vote "q" [a, b]` is a poll, not also a vote
    kept, end = [], -1
    for start, raw, cmd in found:
        if start >= end:
            kept.append((start, raw, cmd))
            end = start + len(raw)
    found = kept
    if any(c.action == "poll" for _, _, c in found):
        for _, _, c in found:
            if c.action == "poll":
                c.args["options"] = [o.strip() for o in c.args["options"] if o.strip()][:8]
    cleaned = text
    for _, raw, _ in found:
        cleaned = cleaned.replace(raw, "", 1)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, [c for _, _, c in found]


_PASS_WORDS = ("pass", "[pass]", "(pass)", "*pass*", "!pass", "pass.")


def is_pass(text):
    return text.strip().lower() in _PASS_WORDS


def could_be_pass(text):
    """True while streamed text might still turn out to be a pass."""
    t = text.strip().lower()
    return not t or any(word.startswith(t) for word in _PASS_WORDS)


def strip_name_prefix(text, names):
    """Models sometimes echo the transcript format: "[Opus]: hi" -> "hi"."""
    for name in names:
        for prefix in (f"[{name}]:", f"{name}:", f"**{name}**:", f"[{name}]"):
            if text.lstrip().lower().startswith(prefix.lower()):
                return text.lstrip()[len(prefix):].lstrip()
    return text
