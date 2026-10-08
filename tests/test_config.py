"""Settings: defaults, precedence (code > env > toml > defaults), per-question sections, explain, validation."""
import pytest

from shad0w import config


def test_defaults():
    s = config.resolve()
    assert s.alpha == 0.05 and s.mode == "serve" and s.canary == 1.0 and s.never_serve == () and s.auto_train == 0
    assert set(s.asdict()) == set(config.DEFAULTS)


def test_precedence_toml_env_code(tmp_path, monkeypatch):
    (tmp_path / "shad0w.toml").write_text('alpha = 0.02\nmode = "shadow"\nnever_serve = ["fraud"]\n'
                                          '[questions.intent]\nalpha = 0.01\ncanary = 0.5\n')
    monkeypatch.chdir(tmp_path)
    s = config.resolve()
    assert s.alpha == 0.02 and s.mode == "shadow" and s.never_serve == ("fraud",)
    q = config.resolve("intent")
    assert q.alpha == 0.01 and q.canary == 0.5 and q.mode == "shadow"
    assert config.resolve("other").alpha == 0.02
    monkeypatch.setenv("SHAD0W_ALPHA", "0.03")
    monkeypatch.setenv("SHAD0W_NEVER_SERVE", "a, b")
    monkeypatch.setenv("SHAD0W_AUTO_TRAIN", "off")
    monkeypatch.setenv("SHAD0W_EXPOSED", "true")
    s = config.resolve("intent")
    assert s.alpha == 0.03 and s.never_serve == ("a", "b") and s.auto_train == 0 and s.exposed is True
    assert config.resolve("intent", alpha=0.04).alpha == 0.04
    assert config.resolve("intent", alpha=None).alpha == 0.03  # None means "not given"
    rows = {k: (v, src) for k, v, src in config.explain("intent", alpha=0.04)}
    assert rows["alpha"] == (0.04, "code") and rows["never_serve"][1] == "env:SHAD0W_NEVER_SERVE"
    assert rows["canary"][1].endswith("[questions.intent]") and rows["delta"][1] == "default"


def test_config_path_env(tmp_path, monkeypatch):
    p = tmp_path / "custom.toml"
    p.write_text("audit_rate = 0.2\n")
    monkeypatch.setenv("SHAD0W_CONFIG", str(p))
    assert config.resolve().audit_rate == 0.2
    monkeypatch.setenv("SHAD0W_CONFIG", str(tmp_path / "missing.toml"))
    with pytest.raises(FileNotFoundError):
        config.resolve()


@pytest.mark.parametrize("bad", [{"mode": "maybe"}, {"canary": 1.5}, {"audit_rate": -1}, {"retrain": "x"},
                                 {"min_rows": 10}, {"capture": ["header", "magic"]}, {"nope": 1}])
def test_validation(bad):
    with pytest.raises((ValueError, TypeError)):
        config.resolve(**bad)


def test_force_threshold_warns(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="shad0w"):
        s = config.resolve(force_threshold=0.5)
    assert s.force_threshold == 0.5 and "certificate" in caplog.text
