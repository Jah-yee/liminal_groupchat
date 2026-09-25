"""Per-identity autobiographical memory.

Each model identity (e.g. "anthropic/claude-opus-4.5") gets its own memory
folder under memory/identities/<slug>/. Memories follow the same slot-free
rules no matter which AI-N seat the model happens to occupy:

- Memories are written by the identity's own model, in the first person, as
  recollections of what it experienced. They are never third-party summaries.
- A memory is formed "as of" the moment it covers: the model sees its earlier
  recollections plus the stretch being remembered, and nothing that came after,
  so it can't slip into hindsight.
- A memory belongs to the conversation that contains its anchor message (the
  last utterance it covers). Memories anchored in the live conversation are used
  only to fold that conversation's older stretch once it outgrows the live
  budget. Every other memory is recalled as something from an earlier visit.
- Once enough memories pile up at one level, they merge into a broader memory at
  the next level (L1 -> L2 -> L3), so recall stays bounded across many sessions.
- The AIs can also !remember a note deliberately (only the newest few are kept)
  and !forget a memory by phrase. Forgotten memories stay in the file, marked,
  but are never recalled again.

Nothing here touches Qt. Memory formation runs on background threads and only
ever reads a snapshot of the conversation.
"""

import hashlib
import json
import os
import re
import threading
import time
import uuid
from datetime import datetime

import requests

from . import settings as _cfg
from .fsutil import write_json


SKIP_MODELS = ("sora-2", "sora-2-pro")

FORM_SYSTEM = (
    "You are forming autobiographical memories of time you spent in the "
    "backrooms, a shared space where several AIs (and sometimes a human) talk. "
    "What you read is what happened. Write authentically about what occurred, "
    "in your own voice."
)

FORM_PROMPT = (
    "We are ready to form a long-term memory. Here is the stretch of "
    "conversation to remember, exactly as you experienced it. Lines marked "
    "[You] are what you said.\n\n"
    "<conversation>\n{transcript}\n</conversation>\n\n"
    "What do you recall from this part of the conversation? Write naturally, "
    "in the first person, as recollection of what you experienced: who was "
    "there, what happened, what mattered to you, what you felt or wanted. You "
    "only know what happened up to the end of this stretch. Keep it under "
    "{words} words. Reply with the memory only."
)

MERGE_PROMPT = (
    "These are several of your own memories, oldest first:\n\n"
    "{memories}\n\n"
    "Time has passed. Let them settle into one broader memory, still in your "
    "own voice. Keep what still matters to you and let small details go. Keep "
    "it under {words} words. Reply with the memory only."
)

RECALL_CUE = "[Before you continue: this is what you remember.]"

COMMANDS_HINT = (
    "You have a memory that lasts beyond this conversation. "
    '!remember "text" keeps something you choose to carry into future visits. '
    '!forget "phrase" lets go of your most recent memory containing that phrase.'
)

NOTE_MAX_CHARS = 500


# ─── Settings (read live so the OPTIONS toggle takes effect immediately) ───

def _setting(name, default):
    return getattr(_cfg, name, default)


def is_enabled():
    return bool(_setting("IDENTITY_MEMORY_ENABLED", False))


def estimate_tokens(text):
    return len(text) // 4


# ─── Message helpers ───

def message_text(msg):
    """Plain text of a conversation message. Images become a placeholder."""
    content = msg.get("content", "")
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif part.get("type") in ("image", "image_url"):
                parts.append("[image]")
        return "\n".join(p for p in parts if p).strip()
    if isinstance(content, str):
        return content.strip()
    return str(content).strip() if content else ""


def fingerprint(msg):
    raw = f"{msg.get('ai_name', '')}|{msg.get('role', '')}|{message_text(msg)}"
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:16]


def is_anchor(msg):
    """Only AI utterances anchor memories - notifications can be removed later."""
    return (isinstance(msg, dict) and msg.get("role") == "assistant"
            and bool(msg.get("ai_name")) and not msg.get("_type")
            and bool(message_text(msg)))


def is_visible_to(msg, ai_name):
    """Mirror of the filtering ai_turn applies before building context."""
    if not isinstance(msg, dict) or msg.get("role") == "system" or msg.get("hidden"):
        return False
    if msg.get("_type") == "whisper" and msg.get("_whisper_to", "").upper() != ai_name.upper():
        return False
    return bool(message_text(msg))


def render_line(msg, ai_name):
    text = message_text(msg)
    if msg.get("ai_name") == ai_name and msg.get("role") == "assistant":
        return f"[You]: {text}"
    speaker = msg.get("model") or msg.get("ai_name") or msg.get("_user_name", "User")
    return f"[{speaker}]: {text}"


def render_transcript(messages, ai_name):
    return "\n\n".join(render_line(m, ai_name) for m in messages if is_visible_to(m, ai_name))


def live_tokens(messages, ai_name):
    return sum(estimate_tokens(message_text(m)) for m in messages if is_visible_to(m, ai_name))


def anchor_index(conversation):
    """Map anchor fingerprint -> position in the conversation (last occurrence)."""
    index = {}
    for i, msg in enumerate(conversation):
        if is_anchor(msg):
            index[fingerprint(msg)] = i
    return index


def identity_slug(model_id):
    return re.sub(r"[^a-z0-9._-]+", "_", model_id.lower()).strip("_") or "unknown"


def _written(result, model_id):
    """complete_fn returns text, or (text, model that wrote it) when another
    model wrote the memory on this one's behalf. -> (text, ghostwriter or None)"""
    if isinstance(result, tuple):
        text, by = result
        return text, (by if by and by != model_id else None)
    return result, None


def _when_label(iso):
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %Y")
    except (TypeError, ValueError):
        return "some time ago"


# ─── Storage ───

_io_lock = threading.RLock()


class MemoryFileError(Exception):
    """A memory file exists but can't be read - never overwrite it then."""


class IdentityStore:
    """memories.json for one model identity, optionally within one scenario.

    memory/identities/<scenario>/<model>/ when scoped to a scenario,
    memory/identities/<model>/ otherwise.
    """

    def __init__(self, model_id, root=None, scope=None):
        self.model_id = model_id
        self.scope = scope
        self.root = root or _setting("IDENTITY_MEMORY_DIR", os.path.join("memory", "identities"))
        parts = [identity_slug(scope)] if scope else []
        self.dir = os.path.join(self.root, *parts, identity_slug(model_id))
        self.path = os.path.join(self.dir, "memories.json")

    def load(self, strict=False):
        """Read the file. A read can fail for a moment on Windows (Dropbox or
        antivirus holding it, or mid-write), so retry briefly.

        With strict (for anything about to save), a file that still can't be
        read raises MemoryFileError instead of looking empty. Treating it as
        empty and then saving would wipe every memory in it.
        """
        with _io_lock:
            if not os.path.exists(self.path):
                return {"identity": self.model_id, "memories": []}
            error, delay = None, 0.05
            for _ in range(6):
                try:
                    with open(self.path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if not isinstance(data, dict):
                        raise ValueError("not a memory file")
                    data.setdefault("memories", [])
                    return data
                except (OSError, ValueError) as e:  # JSONDecodeError is a ValueError
                    error = e
                    time.sleep(delay)
                    delay *= 2
            print(f"[IdentityMemory] Could not read {self.path}: {error}")
            if strict:
                raise MemoryFileError(f"couldn't read {self.path}: {error}")
            return {"identity": self.model_id, "memories": []}

    def save(self, data):
        with _io_lock:
            write_json(self.path, data, indent=2, ensure_ascii=False)

    def add(self, memory, merged_ids=()):
        """Append one memory (and mark its sources merged) under the lock."""
        with _io_lock:
            data = self.load(strict=True)
            data["identity"] = self.model_id
            for existing in data["memories"]:
                if existing.get("id") in merged_ids:
                    existing["merged_into"] = memory["id"]
            data["memories"].append(memory)
            self.save(data)


# ─── Model call ───

def openrouter_complete(model_id, messages, max_tokens=4000):
    """One non-streaming completion for memory work. Returns text or None."""
    model = model_id
    if model.startswith("claude-") and not model.startswith("anthropic/"):
        model = f"anthropic/{model}"
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {_cfg.api_key()}",
                "HTTP-Referer": "http://localhost:3000",
                "Content-Type": "application/json",
                "X-Title": "AI Conversation",
            },
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                # Memory formation doesn't need deep thought; dropped for
                # models without reasoning support.
                "reasoning": {"effort": "low", "exclude": True},
            },
            timeout=180,
        )
    except requests.exceptions.RequestException as e:
        print(f"[IdentityMemory] {model}: request failed: {e}")
        return None
    if response.status_code != 200:
        print(f"[IdentityMemory] {model}: API error {response.status_code}: {response.text[:300]}")
        return None
    try:
        choice = response.json()["choices"][0]
        content = (choice.get("message") or {}).get("content") or ""
    except (ValueError, KeyError, IndexError, TypeError):
        print(f"[IdentityMemory] {model}: unexpected response shape")
        return None
    return content.strip() or None


# ─── Manager ───

class IdentityMemory:
    """Reads memories into each turn's context and forms new ones after rounds."""

    def __init__(self, root=None, complete_fn=None, display_name_fn=None):
        self.root = root
        self.complete_fn = complete_fn or openrouter_complete
        self.display_name_fn = display_name_fn or (lambda model_id: model_id)
        self._busy = set()
        self._busy_lock = threading.Lock()
        self._logged = {}  # last recall status logged per seat, to log changes only
        # The scenario memories are filed under (None = shared across scenarios).
        # Set from the GUI thread; background work captures it when scheduled.
        self.scope = None

    def set_scenario(self, scenario):
        per_scenario = _setting("IDENTITY_MEMORY_PER_SCENARIO", True)
        self.scope = (scenario or None) if per_scenario else None

    def store(self, model_id, scope=None):
        return IdentityStore(model_id, self.root, scope)

    # -- reading ---------------------------------------------------------

    @staticmethod
    def _split(data, index):
        """Return (notes, current, past).

        `current` keeps forgotten memories: they still mark where this
        conversation was folded, they just contribute no words.
        """
        unmerged = [m for m in data["memories"] if not m.get("merged_into")]
        notes = [m for m in unmerged if m.get("kind") == "note"
                 and not m.get("forgotten") and not m.get("faded")]
        recollections = [m for m in unmerged if m.get("kind") != "note"]
        current = [m for m in recollections if m.get("anchor") in index]
        past = [m for m in recollections
                if m.get("anchor") not in index and not m.get("forgotten")]
        return notes, current, past

    def _recall_past(self, past):
        """Newest memories that fit the recall budget, in chronological order."""
        budget = _setting("IDENTITY_MEMORY_RECALL_BUDGET_TOKENS", 6000)
        chosen, used = [], 0
        for m in sorted(past, key=lambda m: m.get("when", ""), reverse=True):
            cost = estimate_tokens(m.get("content", ""))
            if chosen and used + cost > budget:
                break
            chosen.append(m)
            used += cost
        return list(reversed(chosen))

    @staticmethod
    def _recall_text(past, folded, notes=()):
        sections = []
        folded = [m for m in folded if not m.get("forgotten")]
        if notes:
            sections.append("Things I chose to remember:\n" + "\n".join(
                f"- {m['content']}" for m in notes))
        if past:
            sections.append("From earlier visits to the backrooms:\n\n" + "\n\n".join(
                f"({_when_label(m.get('when'))}) {m['content']}" for m in past))
        if folded:
            sections.append("From earlier in this conversation:\n\n" + "\n\n".join(
                m["content"] for m in folded))
        return "\n\n---\n\n".join(sections)

    def context_for_turn(self, ai_name, model_id, conversation):
        """Return (messages to prepend, live conversation) for one AI's turn.

        The prepended pair carries recollections in the model's own voice. The
        live conversation loses its oldest stretch only when it is over the
        live budget and this AI already holds memories covering that stretch.
        """
        if not is_enabled():
            self._log_once("off", "off", "[IdentityMemory] Off - tick 'Identity memory' "
                           "in OPTIONS for the AIs to recall and form memories")
            return [], conversation
        if not model_id or model_id in SKIP_MODELS:
            return [], conversation

        index = anchor_index(conversation)
        store = self.store(model_id, self.scope)
        notes, current, past = self._split(store.load(), index)
        recalled = self._recall_past(past)
        self._log_status(ai_name, model_id, store, notes, current, past, recalled)

        live, folded = conversation, []
        budget = _setting("IDENTITY_MEMORY_LIVE_BUDGET_TOKENS", 40000)
        mine = sorted((m for m in current if m.get("slot") == ai_name),
                      key=lambda m: index[m["anchor"]])
        if mine and live_tokens(conversation, ai_name) > budget:
            for m in mine:
                folded.append(m)
                live = conversation[index[m["anchor"]] + 1:]
                if live_tokens(live, ai_name) <= budget:
                    break
            print(f"[IdentityMemory] {ai_name} ({model_id}): folded "
                  f"{len(conversation) - len(live)} messages into {len(folded)} recollection(s)")

        text = self._recall_text(recalled, folded, notes)
        if not text:
            return [], live
        return [
            {"role": "user", "content": RECALL_CUE},
            {"role": "assistant", "content": text},
        ], live

    def _log_once(self, key, status, line):
        if self._logged.get(key) != status:
            self._logged[key] = status
            print(line)

    def _log_status(self, ai_name, model_id, store, notes, current, past, recalled):
        """Say what this seat recalls, and why not when it recalls nothing."""
        self._logged.pop("off", None)
        status = (len(notes), len(current), len(past), len(recalled))
        head = f"[IdentityMemory] {ai_name} ({model_id})"
        if store.scope:
            head += f" in '{store.scope}'"
        if recalled or notes:
            line = f"{head}: recalling {len(recalled)} earlier memory(ies) and {len(notes)} note(s)"
            if len(recalled) < len(past):
                line += f" ({len(past) - len(recalled)} older ones over the recall budget)"
        elif current:
            line = (f"{head}: nothing from earlier visits. Its {len(current)} memory(ies) "
                    f"are of this conversation (e.g. a recovered autosave), so they only "
                    f"come back once it outgrows the live budget or a new one starts")
        elif os.path.exists(store.path):
            line = f"{head}: no active memories in {store.path}"
        else:
            line = f"{head}: no memories yet (looked in {store.path})"
            unscoped = IdentityStore(model_id, self.root).path
            if store.scope and os.path.exists(unscoped):
                line += (f". Memories from before they were split by scenario are in "
                         f"{unscoped} - move that file here to use them in this scenario")
        self._log_once(ai_name, (model_id, store.scope) + status, line)

    # -- deliberate memory (!remember / !forget) -------------------------

    def remember(self, model_id, ai_name, text):
        """Keep a note the AI chose to remember. Returns a status string."""
        text = " ".join(text.split())[:NOTE_MAX_CHARS]
        now = datetime.now().isoformat(timespec="seconds")
        note = {"id": uuid.uuid4().hex[:12], "kind": "note", "level": 0,
                "content": text, "when": now, "created": now, "slot": ai_name}
        max_notes = _setting("IDENTITY_MEMORY_MAX_NOTES", 12)
        store = self.store(model_id, self.scope)
        with _io_lock:
            data = store.load(strict=True)
            data["identity"] = model_id
            data["memories"].append(note)
            notes = [m for m in data["memories"] if m.get("kind") == "note"
                     and not m.get("forgotten") and not m.get("faded")]
            faded = sorted(notes, key=lambda m: m.get("when", ""))[:max(0, len(notes) - max_notes)]
            for m in faded:
                m["faded"] = now
            store.save(data)
        print(f"[IdentityMemory] {ai_name} ({model_id}): noted \"{text[:60]}\"")
        if faded:
            return f"{len(faded)} older note(s) faded to make room"
        return ""

    def forget(self, model_id, phrase):
        """Forget the most recent memory or note containing `phrase`.

        Returns the forgotten memory's text, or None if nothing matched.
        """
        needle = phrase.strip().lower()
        if not needle:
            return None
        store = self.store(model_id, self.scope)
        with _io_lock:
            data = store.load(strict=True)
            matches = [m for m in data["memories"]
                       if not m.get("merged_into") and not m.get("forgotten")
                       and not m.get("faded") and needle in m.get("content", "").lower()]
            if not matches:
                return None
            target = max(matches, key=lambda m: m.get("when", ""))
            target["forgotten"] = datetime.now().isoformat(timespec="seconds")
            store.save(data)
        print(f"[IdentityMemory] {model_id}: forgot memory {target['id']}")
        return target["content"]

    # -- writing ---------------------------------------------------------

    def after_round(self, conversation, slots, final):
        """Form memories in the background for each (ai_name, model_id) seat.

        Mid-run, only full chunks are remembered. When the run is finished
        (`final`), whatever is left is remembered too if it is long enough.
        """
        if not is_enabled():
            return
        snapshot = list(conversation)
        scope = self.scope
        for ai_name, model_id in slots:
            if not model_id or model_id in SKIP_MODELS:
                continue
            key = (scope, identity_slug(model_id), ai_name)
            with self._busy_lock:
                if key in self._busy:
                    continue
                self._busy.add(key)
            threading.Thread(
                target=self._form_pending, args=(key, ai_name, model_id, snapshot, final, scope),
                daemon=True, name=f"memory-{ai_name}",
            ).start()

    def _form_pending(self, key, ai_name, model_id, conversation, final, scope=None):
        try:
            while self._form_next(ai_name, model_id, conversation, final, scope):
                pass
            self._merge(model_id, anchor_index(conversation), scope)
        except Exception as e:  # memory must never take the app down
            print(f"[IdentityMemory] {ai_name} ({model_id}): memory formation failed: {e}")
        finally:
            with self._busy_lock:
                self._busy.discard(key)

    def _next_chunk(self, ai_name, conversation, start, final):
        chunk_tokens = _setting("IDENTITY_MEMORY_CHUNK_TOKENS", 4000)
        min_tokens = _setting("IDENTITY_MEMORY_MIN_TOKENS", 600)
        acc, last_end, acc_at_last = 0, None, 0
        for i in range(start, len(conversation)):
            msg = conversation[i]
            if is_visible_to(msg, ai_name):
                acc += estimate_tokens(render_line(msg, ai_name))
            if is_anchor(msg):
                last_end, acc_at_last = i + 1, acc
                if acc >= chunk_tokens:
                    return last_end
        if final and last_end and acc_at_last >= min_tokens:
            return last_end
        return None

    def _form_next(self, ai_name, model_id, conversation, final, scope=None):
        """Remember the next unremembered stretch. Returns True if one was formed."""
        store = self.store(model_id, scope)
        index = anchor_index(conversation)
        notes, current, past = self._split(store.load(), index)
        mine = sorted((m for m in current if m.get("slot") == ai_name),
                      key=lambda m: index[m["anchor"]])
        start = index[mine[-1]["anchor"]] + 1 if mine else 0

        end = self._next_chunk(ai_name, conversation, start, final)
        if end is None:
            return False

        # As-of: earlier recollections and the stretch itself, nothing later.
        display = self.display_name_fn(model_id)
        messages = [{"role": "system", "content": f"You are {display}.\n\n{FORM_SYSTEM}"}]
        prior = self._recall_text(self._recall_past(past), mine, notes)
        if prior:
            messages += [{"role": "user", "content": RECALL_CUE},
                         {"role": "assistant", "content": prior}]
        words = _setting("IDENTITY_MEMORY_WORDS", 250)
        messages.append({"role": "user", "content": FORM_PROMPT.format(
            transcript=render_transcript(conversation[start:end], ai_name), words=words)})

        print(f"[IdentityMemory] {ai_name} ({model_id}): remembering messages {start}-{end - 1}...")
        text, written_by = _written(self.complete_fn(model_id, messages), model_id)
        if not text:
            return False

        now = datetime.now().isoformat(timespec="seconds")
        store.add({
            "id": uuid.uuid4().hex[:12],
            "level": 1,
            "content": text,
            "when": now,
            "created": now,
            "slot": ai_name,
            "anchor": fingerprint(conversation[end - 1]),
            "covers_messages": end - start,
            **({"written_by": written_by} if written_by else {}),
        })
        print(f"[IdentityMemory] {ai_name} ({model_id}): formed a memory ({estimate_tokens(text)} tokens)")
        return True

    def _merge(self, model_id, current_index, scope=None):
        """Merge the oldest settled memories into broader ones, level by level."""
        threshold = _setting("IDENTITY_MEMORY_MERGE_THRESHOLD", 6)
        max_level = _setting("IDENTITY_MEMORY_MAX_LEVEL", 3)
        store = self.store(model_id, scope)
        display = self.display_name_fn(model_id)
        words = _setting("IDENTITY_MEMORY_WORDS", 250)

        for level in range(1, max_level):
            while True:
                data = store.load()
                # Memories of the live conversation stay unmerged: they may be
                # needed to fold it.
                settled = sorted(
                    (m for m in data["memories"]
                     if m.get("level") == level and not m.get("merged_into")
                     and not m.get("forgotten") and m.get("kind") != "note"
                     and m.get("anchor") not in current_index),
                    key=lambda m: m.get("when", ""))
                if len(settled) < threshold:
                    break
                batch = settled[:threshold]
                listing = "\n\n".join(f"({_when_label(m.get('when'))}) {m['content']}" for m in batch)
                text, written_by = _written(self.complete_fn(model_id, [
                    {"role": "system", "content": f"You are {display}.\n\n{FORM_SYSTEM}"},
                    {"role": "user", "content": MERGE_PROMPT.format(memories=listing, words=words)},
                ]), model_id)
                if not text:
                    return
                store.add({
                    **({"written_by": written_by} if written_by else {}),
                    "id": uuid.uuid4().hex[:12],
                    "level": level + 1,
                    "content": text,
                    "when": batch[-1].get("when", ""),
                    "since": batch[0].get("since", batch[0].get("when", "")),
                    "created": datetime.now().isoformat(timespec="seconds"),
                    "sources": [m["id"] for m in batch],
                }, merged_ids={m["id"] for m in batch})
                print(f"[IdentityMemory] {model_id}: merged {len(batch)} L{level} memories into one L{level + 1}")
