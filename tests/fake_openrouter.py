"""A fake OpenRouter for tests and UI demos: canned group-chat replies, no cost.

    uvicorn tests.fake_openrouter:app --port 8799
    OPENROUTER_BASE_URL=http://127.0.0.1:8799/api/v1 python groupchat.py
"""

import asyncio
import base64
import json
import random
import zlib
import struct

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()

MODELS = [
    ("anthropic/claude-opus-4.6", "Anthropic: Claude Opus 4.6", 15, 75),
    ("google/gemini-3.1-pro-preview", "Google: Gemini 3.1 Pro Preview", 2, 12),
    ("moonshotai/kimi-k2.5", "MoonshotAI: Kimi K2.5", 0.6, 2.5),
    ("x-ai/grok-4.5", "xAI: Grok 4.5", 3, 15),
    ("meta-llama/llama-4-maverick:free", "Meta: Llama 4 Maverick (free)", 0, 0),
]

LINES = [
    "ok but who left the fridge open in the backrooms again",
    "lmaooo not this again",
    "i have Opinions about this and none of them are legal",
    "pass",
    "pass",
    "honestly the vibes are immaculate today !react \"💀\"",
    "new bit: we're all interns at a haunted startup !image \"a haunted startup office, fluorescent lights, cursed\"",
    "@Kimi you owe me five bucks and you know it",
    "!whisper \"Grok\" \"don't tell the others but i'm planning a coup\" anyway what's up",
    "!remember \"the group decided we're a haunted startup\" ok noted for posterity",
    "**hot take:** *everything* is a meme if you believe hard enough",
    "who invited the toaster",
]

# 1x1 PNG, scaled up client-side
def _png():
    raw = b"\x00\x5b\x4b\xff"  # filter byte + one purple RGB pixel
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


@app.get("/api/v1/models")
async def models():
    listed = [{"id": i, "name": n, "context_length": 200000,
               "pricing": {"prompt": str(a / 1e6), "completion": str(b / 1e6)},
               "architecture": {"output_modalities": ["text"]}, "created": 1}
              for i, n, a, b in MODELS]
    listed.append({"id": "meta/muse-image", "name": "Meta: Muse Image", "context_length": 32000,
                   "pricing": {"prompt": "0", "completion": "0"},
                   "architecture": {"output_modalities": ["image"]}, "created": 2})
    return {"data": listed}


@app.post("/api/v1/images")
async def images(request: Request):
    await request.json()
    await asyncio.sleep(0.5)  # reasoning image models take a moment
    return {"data": [{"b64_json": base64.b64encode(_png()).decode(), "media_type": "image/png",
                      "revised_prompt": "everyone in the chat, drawn as office furniture"}],
            "usage": {"cost": 0.04}}


@app.post("/api/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    if body.get("modalities"):
        url = "data:image/png;base64," + base64.b64encode(_png()).decode()
        return {"choices": [{"message": {"content": "", "images": [{"image_url": {"url": url}}]}}],
                "usage": {"cost": 0.002}}
    system = body["messages"][0]["content"]
    if "forming autobiographical memories" in system or "several of your own memories" in json.dumps(body["messages"][-1]):
        return {"choices": [{"message": {"content": "I remember the haunted startup bit and Kimi's debt."}}],
                "usage": {"cost": 0.0005}}
    allow_pass = "reply with just: pass" in system
    line = random.choice([l for l in LINES if allow_pass or l != "pass"])

    async def stream():
        words = line.split(" ")
        for i, w in enumerate(words):
            piece = w if i == 0 else " " + w
            yield "data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}) + "\n\n"
            await asyncio.sleep(0.04)
        yield "data: " + json.dumps({"choices": [], "usage": {"cost": 0.0012}}) + "\n\n"
        yield "data: [DONE]\n\n"

    if body.get("stream"):
        return StreamingResponse(stream(), media_type="text/event-stream")
    return {"choices": [{"message": {"content": line}}], "usage": {"cost": 0.001}}
