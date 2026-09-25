#!/usr/bin/env python3
"""Start Liminal Groupchat and open it in the browser.

    python groupchat.py            # http://127.0.0.1:8765
    python groupchat.py --port 9000 --no-browser
"""

import argparse
import os
import threading
import webbrowser

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Liminal Groupchat")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        # Reachable from other devices: allow their Host names (Origin is still checked)
        os.environ["GROUPCHAT_LAN"] = "1"
        print("\n  ⚠  Listening beyond this computer. There's no login: anyone who can reach"
              "\n     this address can use the app, and spend your OpenRouter credits.")
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}"
    print(f"\n  Liminal Groupchat → {url}\n")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run("gchat.server:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
