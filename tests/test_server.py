import json
import logging
import threading
from http.client import HTTPConnection

import pytest

from server import BridgeHTTPServer, LogBuffer
from vrcai.config import ConfigStore
from vrcai.service import BridgeController


@pytest.fixture
def web(tmp_path, cfg):
    store = ConfigStore(tmp_path / "config.yaml")
    store.update(cfg)
    server = BridgeHTTPServer(("127.0.0.1", 0), BridgeController(store))
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.02))
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(2)
    assert not thread.is_alive()


def request(web, path, data=None, *, token=True, headers=None, raw=None):
    conn = HTTPConnection("127.0.0.1", web.server_address[1], timeout=3)
    fields = {"X-Session-Token": web.token} if token else {}
    fields.update(headers or {})
    body = raw if raw is not None else json.dumps(data) if data is not None else None
    if body is not None:
        fields.setdefault("Content-Type", "application/json")
    conn.request("POST" if body is not None else "GET", path, body=body, headers=fields)
    response = conn.getresponse()
    status, result, response_headers = response.status, response.read(), dict(response.getheaders())
    conn.close()
    return status, result, response_headers


def test_page_injects_session_and_uses_external_script(web):
    status, page, headers = request(web, "/", token=False)
    assert status == 200
    assert web.token.encode() in page
    assert b"__SESSION_TOKEN__" not in page
    assert headers["X-Frame-Options"] == "DENY"
    assert "script-src 'self'" in headers["Content-Security-Policy"]
    assert request(web, "/app.js", token=False)[0] == 200


def test_config_requires_auth_and_never_returns_secret(web):
    assert request(web, "/api/config", token=False)[0] == 403
    status, content, headers = request(web, "/api/config")
    assert status == 200
    assert b"test-llm-secret" not in content and b"test-stt-secret" not in content
    assert json.loads(content)["stt"]["groq_whisper"]["key_configured"]
    assert "Access-Control-Allow-Origin" not in headers


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://evil.example"},
        {"Host": "evil.example:8765"},
        {"Origin": "null"},
        {"Sec-Fetch-Site": "cross-site"},
    ],
)
def test_host_origin_and_fetch_site_guards_prevent_cross_site_reads_and_writes(web, headers):
    assert request(web, "/api/config", headers=headers)[0] == 403
    assert request(web, "/api/config", {"system_prompt": "attacker"}, headers=headers)[0] == 403
    assert web.controller.store.load()["system_prompt"] != "attacker"


def test_authenticated_patch_preserves_other_settings(web):
    status, _, _ = request(web, "/api/config", {"chat": {"temperature": 0}})
    assert status == 200
    cfg = json.loads(request(web, "/api/config")[1])
    assert cfg["chat"]["temperature"] == 0
    assert cfg["llm"]["openai_compatible"]["base_url"] == "https://api.deepseek.com/v1"
    assert web.controller.store.load()["llm"]["openai_compatible"]["api_key"] == "test-llm-secret"


@pytest.mark.parametrize(
    "raw",
    ['{"chat":{"temperature":NaN}}', "[]", "{broken", '{"tts":true}', "x" * 65537],
    ids=["nonfinite", "array", "syntax", "schema", "oversize"],
)
def test_invalid_json_and_schema_return_400_without_writing(web, raw):
    before = web.controller.store.path.read_bytes()
    assert request(web, "/api/config", raw=raw)[0] == 400
    assert web.controller.store.path.read_bytes() == before


def test_disconnected_actions_return_clear_errors(web):
    status, body, _ = request(web, "/api/message", {"text": "你好"})
    assert status == 400
    assert "请先启动" in json.loads(body)["error"]
    assert request(web, "/api/listen", {"active": "true"})[0] == 400


def test_logs_are_bounded_and_independent_for_multiple_clients():
    logs = LogBuffer(capacity=3)
    for i in range(5):
        logs.emit(logging.LogRecord("vrcai", logging.INFO, "", 0, "message %d", (i,), None))
    first, second = logs.read(0), logs.read(0)
    assert first == second
    assert len(first["logs"]) == 3
    assert first["cursor"] == 5
    assert logs.read(5)["logs"] == []
