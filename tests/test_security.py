"""The local server refuses other websites. Run: python -m pytest tests"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GROUPCHAT_DATA_DIR", tempfile.mkdtemp())

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from gchat import settings, store  # noqa: E402
from gchat.server import app  # noqa: E402

HERE = "http://localhost:8765"


@pytest.fixture()
def client():
    os.environ.pop("GROUPCHAT_LAN", None)
    with TestClient(app, base_url=HERE) as c:
        yield c


def test_own_page_works(client):
    assert client.get("/").status_code == 200
    r = client.post("/api/control/pause", headers={"Origin": HERE})
    assert r.status_code == 200


def test_other_websites_cannot_press_play(client):
    r = client.post("/api/control/play", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    r = client.post("/api/control/play", headers={"Origin": "null"})  # sandboxed iframe
    assert r.status_code == 403


def test_other_websites_cannot_read_chats_over_websocket(client):
    # (the test client sends Host: testserver for WebSockets, so set it)
    local = {"Host": "localhost:8765"}
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={**local, "Origin": "https://evil.example"}) as ws:
            ws.receive_json()
    with client.websocket_connect("/ws", headers={**local, "Origin": HERE}) as ws:
        assert ws.receive_json()["type"] == "snapshot"


def test_dns_rebinding_host_is_refused(client):
    r = client.get("/api/state", headers={"Host": "attacker.example:8765"})
    assert r.status_code == 403


def test_lan_mode_allows_other_hosts_but_still_checks_origin(client):
    os.environ["GROUPCHAT_LAN"] = "1"
    try:
        lan = {"Host": "192.168.1.20:8765"}
        assert client.get("/api/state", headers=lan).status_code == 200
        r = client.post("/api/control/pause", headers={**lan, "Origin": "http://192.168.1.20:8765"})
        assert r.status_code == 200
        r = client.post("/api/control/play", headers={**lan, "Origin": "https://evil.example"})
        assert r.status_code == 403
    finally:
        os.environ.pop("GROUPCHAT_LAN", None)


def test_settings_only_accept_user_settings(client):
    before = settings.get("memory_dir")
    r = client.post("/api/settings", headers={"Origin": HERE},
                    json={"username": "bardo", "memory_dir": "/tmp/elsewhere",
                          "avatars": {"x": "../settings.json"}})
    assert r.status_code == 200
    assert settings.get("username") == "bardo"
    assert settings.get("memory_dir") == before
    assert "x" not in (settings.get("avatars") or {})


def test_chat_ids_cannot_escape_the_chats_folder(client):
    with pytest.raises(ValueError):
        store._path("../settings")
    assert store.load("..%2Fsettings") is None
    r = client.post("/api/chats/..%2F..%2Fsettings/open", headers={"Origin": HERE})
    assert r.status_code in (404, 405)
    store.delete("../settings")  # refused quietly
    assert os.path.exists(settings.SETTINGS_PATH)
