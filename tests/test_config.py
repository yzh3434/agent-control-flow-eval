from src.config import load_config


def test_load_config_reads_yaml_and_env(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "model: deepseek-chat\nbase_url: https://api.deepseek.com\n"
        "sample_size: 10\nmax_rounds: 7\nreflexion_trials: 3\nconcurrency: 4\n"
        "easy_max_steps: 2\nhard_min_steps: 5\ntemperature: 0.0\nrequest_timeout: 60\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    cfg = load_config(str(cfg_file))
    assert cfg.model == "deepseek-chat"
    assert cfg.sample_size == 10
    assert cfg.api_key == "sk-test"
    assert cfg.easy_max_steps == 2
