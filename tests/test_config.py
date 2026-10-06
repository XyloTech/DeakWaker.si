import config

def test_env_float_parses_decimal(monkeypatch):
    monkeypatch.setenv("FLOAT_X", "0.25")
    assert config._env_float("FLOAT_X", 0.1) == 0.25

def test_env_float_missing_or_invalid_returns_default(monkeypatch):
    monkeypatch.delenv("FLOAT_X", raising=False)
    assert config._env_float("FLOAT_X", 0.1) == 0.1
    monkeypatch.setenv("FLOAT_X", "junk")
    assert config._env_float("FLOAT_X", 0.1) == 0.1
