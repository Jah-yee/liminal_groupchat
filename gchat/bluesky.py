"""Read-only Bluesky for !bsky: search posts, or read someone's latest.

    !bsky "query"     posts matching the query, newest first
    !bsky "@handle"   that account's latest posts

Reading an account works without logging in. Bluesky only allows search for
logged-in apps, so search uses the handle and app password from Settings
(bsky.app → Settings → Privacy and security → App passwords). The app only
ever reads.
"""

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from . import settings

PUBLIC = "https://public.api.bsky.app/xrpc/"
TIMEOUT = 15
LOGIN_HINT = ("(Bluesky only lets logged-in apps search. Add a Bluesky handle and app "
              "password in Settings - or read someone's posts with !bsky \"@handle\")")

_session = {"key": None, "jwt": None, "pds": None}
_lock = threading.Lock()


class BlueskyError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _request(url, params=None, body=None, token=None):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {"Accept": "application/json", "User-Agent": "liminal-groupchat"}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode("utf-8"))
            message = detail.get("message") or detail.get("error") or e.reason
        except Exception:
            message = e.reason
        raise BlueskyError(e.code, str(message)) from None


def _credentials():
    handle = (settings.get("bsky_handle") or "").strip().lstrip("@")
    password = (settings.get("bsky_app_password") or "").strip()
    return (handle, password) if handle and password else None


def _login(fresh=False):
    """An access token for the account in Settings (kept until it expires)."""
    creds = _credentials()
    if not creds:
        return None, None
    with _lock:
        if not fresh and _session["key"] == creds and _session["jwt"]:
            return _session["jwt"], _session["pds"]
        data = _request("https://bsky.social/xrpc/com.atproto.server.createSession",
                        body={"identifier": creds[0], "password": creds[1]})
        pds = "https://bsky.social/xrpc/"
        for service in (data.get("didDoc") or {}).get("service", []):
            if service.get("id") == "#atproto_pds" and service.get("serviceEndpoint"):
                pds = service["serviceEndpoint"].rstrip("/") + "/xrpc/"
        _session.update(key=creds, jwt=data["accessJwt"], pds=pds)
        return _session["jwt"], pds


def _get(method, params, auth=False):
    if auth:
        token, pds = _login()
        if token:
            try:
                return _request(pds + method, params, token=token)
            except BlueskyError as e:
                if e.status not in (400, 401):  # 400 ExpiredToken / 401: log in again
                    raise
            token, pds = _login(fresh=True)
            return _request(pds + method, params, token=token)
    return _request(PUBLIC + method, params)


def _ago(stamp):
    try:
        t = time.mktime(time.strptime(stamp[:19], "%Y-%m-%dT%H:%M:%S")) - time.timezone
    except (TypeError, ValueError):
        return ""
    s = max(0, time.time() - t)
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if s >= size:
            return f"{int(s // size)}{unit} ago"
    return "just now"


def _line(post):
    author = post.get("author") or {}
    handle = author.get("handle", "?")
    record = post.get("record") or {}
    text = " ".join((record.get("text") or "").split())
    if len(text) > 280:
        text = text[:277] + "..."
    if not text and (post.get("embed") or {}).get("images"):
        text = "[image]"
    rkey = (post.get("uri") or "").rsplit("/", 1)[-1]
    name = author.get("displayName")
    who = f"@{handle}" + (f" ({name})" if name and name != handle else "")
    stats = f"♥{post.get('likeCount', 0)} 🔁{post.get('repostCount', 0)}"
    when = _ago(record.get("createdAt"))
    return f"- {who} · {when} · {stats}: {text} (https://bsky.app/profile/{handle}/post/{rkey})"


def bluesky(query, max_results=5):
    """Return posts as short plain text for the chat."""
    query = (query or "").strip()
    try:
        if query.startswith("@") and " " not in query:
            actor = query.lstrip("@")
            if "." not in actor:
                actor += ".bsky.social"
            data = _get("app.bsky.feed.getAuthorFeed",
                        {"actor": actor, "limit": max_results, "filter": "posts_no_replies"})
            posts = [item["post"] for item in data.get("feed", []) if "reason" not in item]
        else:
            auth = _credentials() is not None
            try:
                data = _get("app.bsky.feed.searchPosts",
                            {"q": query, "limit": max_results, "sort": "latest"}, auth=auth)
            except BlueskyError as e:
                if not auth and e.status in (401, 403):
                    return LOGIN_HINT
                raise
            posts = data.get("posts", [])
    except BlueskyError as e:
        if e.status == 401 and _credentials():
            return "(Bluesky login failed - check the handle and app password in Settings)"
        return f"(Bluesky: {e})"
    except Exception as e:
        return f"(Bluesky failed: {e})"
    if not posts:
        return "(no posts)"
    return "\n".join(_line(p) for p in posts[:max_results])
