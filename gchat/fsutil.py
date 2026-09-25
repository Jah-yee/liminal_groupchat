"""Safe file writes that survive Windows file locks.

Everything is saved by writing a temporary file and swapping it into place,
so a crash never leaves half a file. On Windows the swap fails with "Access is
denied" while another program has the target open - Dropbox, OneDrive and
antivirus scanners all do this briefly. So retry for a moment, and if the file
stays locked, write it in place instead of losing the save.
"""

import json
import os
import time
import uuid

RETRIES = 8  # waits of 25ms, 50ms, ... about 3 seconds in total


def write_json(path, data, **dump_kwargs):
    write_text(path, json.dumps(data, **dump_kwargs))


def write_text(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{uuid.uuid4().hex[:8]}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    delay = 0.025
    for _ in range(RETRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(delay)
            delay *= 2
    # Still locked: overwrite in place (not atomic, but the data isn't lost)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
