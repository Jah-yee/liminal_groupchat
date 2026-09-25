#!/usr/bin/env python3
"""Check which image endpoints and models your OpenRouter key can use.

    python tools/check_images.py                  # tries meta/muse-image and a known-good model
    python tools/check_images.py some/model-id    # tries that model on /images

Uses the key saved in the app's settings (or OPENROUTER_API_KEY) and prints
OpenRouter's full reply for anything that fails. Each successful call
generates one small image, so it costs a few cents.
"""

import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gchat import llm, settings  # noqa: E402


def try_call(label, url, payload):
    print(f"\n── {label}\n   POST {url}  model={payload['model']}")
    try:
        r = httpx.post(url, headers=llm._headers(), json=payload, timeout=300)
    except httpx.HTTPError as e:
        print(f"   request failed: {e}")
        return
    print(f"   status {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        has_image = bool(data.get("data") or (data.get("choices") or [{}])[0].get("message", {}).get("images"))
        print(f"   ok - image returned: {has_image}, cost: {(data.get('usage') or {}).get('cost')}")
    else:
        print(f"   reply: {r.text[:1500]}")


def main():
    settings.load()
    key = settings.api_key()
    if not key:
        sys.exit("No API key found - set it in the app's Settings first.")
    print(f"Using key …{key[-4:]}")

    r = httpx.get(f"{llm.API}/key", headers=llm._headers(), timeout=30)
    print(f"\n── key check: status {r.status_code}\n   {r.text[:600]}")

    models = sys.argv[1:] or ["meta/muse-image"]
    for model in models:
        try_call(f"{model} on /images", f"{llm.API}/images", {"model": model, "prompt": "a small red cube"})
    if not sys.argv[1:]:
        # The same endpoint with a mainstream model: tells endpoint problems from model problems
        try_call("google/gemini-2.5-flash-image on /images", f"{llm.API}/images",
                 {"model": "google/gemini-2.5-flash-image", "prompt": "a small red cube"})


if __name__ == "__main__":
    main()
