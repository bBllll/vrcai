import json

import pytest
import yaml

from vrcai.config import ConfigError, ConfigStore, DEFAULTS, redact


def test_first_run_defaults_and_old_config_migration(tmp_path):
    store = ConfigStore(tmp_path / "config.yaml")
    assert store.load() == DEFAULTS
    store.path.write_text("audio:\n  device: 7\n  chunk_duration: 0.1\n", encoding="utf-8")
    loaded = store.load()
    assert loaded["audio"]["device"] == 7
    assert loaded["audio"]["silence_duration"] == 0.7


def test_config_round_trip_preserves_custom_model_url_and_zero_temperature(tmp_path):
    store = ConfigStore(tmp_path / "config.yaml")
    custom = {
        "llm": {"openai_compatible": {"model": "custom-model", "base_url": "http://localhost:9999/v1"}},
        "chat": {"temperature": 0},
    }
    store.update(custom)
    store.update({"system_prompt": "new persona"})
    loaded = store.load()
    assert loaded["llm"]["openai_compatible"] == {
        **DEFAULTS["llm"]["openai_compatible"],
        **custom["llm"]["openai_compatible"],
    }
    assert loaded["chat"]["temperature"] == 0


def test_secrets_are_redacted_preserved_when_blank_and_explicitly_clearable(tmp_path, cfg):
    store = ConfigStore(tmp_path / "config.yaml")
    store.update(cfg)
    safe = redact(store.load())
    assert "test-llm-secret" not in json.dumps(safe)
    assert safe["llm"]["openai_compatible"]["key_configured"] is True
    store.update({"llm": {"openai_compatible": {"api_key": ""}}})
    assert store.load()["llm"]["openai_compatible"]["api_key"] == "test-llm-secret"
    store.update({"llm": {"openai_compatible": {"api_key": None}}})
    assert store.load()["llm"]["openai_compatible"]["api_key"] == ""


@pytest.mark.parametrize(
    "updates",
    [
        {"audio": None},
        {"tts": True},
        {"chat": {"temperature": float("nan")}},
        {"chat": {"temperature": True}},
        {"audio": {"device": -1}},
        {"chat": {"queue_size": 0}},
        {"llm": {"provider": "unknown"}},
        {"llm": {"provider": "language"}},
        {"llm": {"openai": {"base_url": "file:///tmp/key"}}},
        {"llm": {"openai": {"api_key": 123}}},
        {"osc": {"send_interval": 0}},
        {"streaming": {"split_chars": []}},
        {"unknown": "field"},
    ],
)
def test_invalid_updates_never_replace_valid_file(tmp_path, updates):
    store = ConfigStore(tmp_path / "config.yaml")
    store.update({})
    before = store.path.read_bytes()
    with pytest.raises(ConfigError):
        store.update(updates)
    assert store.path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_atomic_replace_preserves_original(tmp_path, monkeypatch):
    store = ConfigStore(tmp_path / "config.yaml")
    store.update({})
    before = store.path.read_bytes()

    def fail(*args):
        raise OSError("disk write failure")

    monkeypatch.setattr("vrcai.config.os.replace", fail)
    with pytest.raises(OSError):
        store.update({"system_prompt": "new"})
    assert store.path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_legacy_example_remains_loadable(tmp_path):
    from vrcai.config import ROOT

    # Includes the sample_rate written by the old web interface.
    old = yaml.safe_load((ROOT / "config.yaml.example").read_text(encoding="utf-8-sig"))
    old.setdefault("tts", {}).setdefault("minimax", {})["sample_rate"] = 32000
    store = ConfigStore(tmp_path / "config.yaml")
    store.path.write_text(yaml.safe_dump(old), encoding="utf-8")
    assert store.load()["tts"]["minimax"]["sample_rate"] == 32000
