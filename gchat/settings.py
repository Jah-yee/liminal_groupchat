"""User settings and paths.

Everything the app writes lives under data/ (gitignored): settings.json, saved
chats, generated images and memories. Settings are plain module state so the
memory module can read them live, the way it read config.py in the backrooms.
"""

import json
import os
import threading

from .fsutil import write_json

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("GROUPCHAT_DATA_DIR", os.path.join(APP_DIR, "data"))
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")
CHATS_DIR = os.path.join(DATA_DIR, "chats")
MEDIA_DIR = os.path.join(DATA_DIR, "media")
AVATARS_DIR = os.path.join(DATA_DIR, "avatars")

DEFAULTS = {
    "api_key": "",
    "username": "you",
    "image_model": "google/gemini-3.1-flash-lite-image",
    "thinking": "low",            # off | low | medium | high
    "memory_enabled": True,
    "memory_dir": os.path.join(DATA_DIR, "memory"),
    "show_whispers": True,
    "current_chat": None,
    "avatars": {},                # model ID -> picture filename in AVATARS_DIR
    # Writes memories for a model whose provider refuses its memory requests
    "memory_fallback_model": "anthropic/claude-sonnet-5",
    "time_awareness": True,       # tell the AIs the date/time and when time has passed
}

_lock = threading.RLock()
_settings = dict(DEFAULTS)

# ─── Identity memory knobs, read live by gchat.identity_memory ───
IDENTITY_MEMORY_ENABLED = True
IDENTITY_MEMORY_DIR = DEFAULTS["memory_dir"]
IDENTITY_MEMORY_PER_SCENARIO = True
IDENTITY_MEMORY_CHUNK_TOKENS = 4000
IDENTITY_MEMORY_MIN_TOKENS = 600
IDENTITY_MEMORY_LIVE_BUDGET_TOKENS = 40000
IDENTITY_MEMORY_RECALL_BUDGET_TOKENS = 6000
IDENTITY_MEMORY_MERGE_THRESHOLD = 6
IDENTITY_MEMORY_MAX_LEVEL = 3
IDENTITY_MEMORY_WORDS = 250
IDENTITY_MEMORY_MAX_NOTES = 12

# Memories are filed under this scenario name, so a memory folder can be shared
# with the backrooms app's "Group Chat" scenario.
MEMORY_SCENARIO = "Group Chat"


def _sync_module_state():
    global IDENTITY_MEMORY_ENABLED, IDENTITY_MEMORY_DIR
    IDENTITY_MEMORY_ENABLED = bool(_settings.get("memory_enabled"))
    IDENTITY_MEMORY_DIR = _settings.get("memory_dir") or DEFAULTS["memory_dir"]


def load():
    with _lock:
        os.makedirs(DATA_DIR, exist_ok=True)
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                _settings.update({k: v for k, v in saved.items() if k in DEFAULTS})
        except (OSError, ValueError):
            pass
        _sync_module_state()
        return dict(_settings)


def save():
    with _lock:
        write_json(SETTINGS_PATH, _settings, indent=2)


def get(name):
    with _lock:
        return _settings.get(name, DEFAULTS.get(name))


def update(values):
    """Apply known settings from a dict and persist them."""
    with _lock:
        for key, value in values.items():
            if key in DEFAULTS:
                _settings[key] = value
        _sync_module_state()
        save()
        return dict(_settings)


def api_key():
    return get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")


def public():
    """Settings safe to send to the browser (the key is never sent back)."""
    with _lock:
        data = {k: v for k, v in _settings.items() if k != "api_key"}
    key = api_key()
    data["has_key"] = bool(key)
    data["key_hint"] = f"…{key[-4:]}" if len(key) > 8 else ""
    return data
