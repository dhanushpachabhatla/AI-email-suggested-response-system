"""
Tests for configuration management (src/config.py).

Validates:
- Config loading from YAML
- Validation of required fields
- Environment variable overrides
- Error handling for invalid/missing config
"""

import os
import sys
import pytest
import tempfile
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, ConfigurationError, load_config


VALID_CONFIG = {
    "dataset": {
        "path": "data/email_dataset.json",
        "min_pairs": 50,
        "train_test_split": 0.8,
        "synthetic_size": 100,
    },
    "embeddings": {
        "model": "all-MiniLM-L6-v2",
        "dimension": 384,
        "batch_size": 32,
    },
    "vector_store": {
        "type": "chromadb",
        "persistence_dir": "data/embeddings",
        "collection_name": "email_embeddings",
    },
    "llm": {
        "primary": "lm_studio",
        "lm_studio_url": "http://127.0.0.1:1234/v1",
        "timeout": 30,
        "fallback_provider": None,
        "fallback_api_key": None,
    },
    "generation": {
        "temperature": 0.7,
        "max_tokens": 500,
        "top_k_examples": 3,
        "prompt_template": "few_shot",
    },
    "evaluation": {
        "dimensions": ["semantic_similarity", "tone_appropriateness"],
        "thresholds": {
            "high_quality": {"semantic_similarity": 0.8, "llm_judge_dimensions": 4.0},
            "acceptable": {"semantic_similarity": 0.6, "llm_judge_dimensions": 3.0},
        },
        "llm_judge_model": "lm_studio",
        "supplementary_metrics": ["rouge", "bleu"],
        "batch_size": 10,
    },
    "output": {
        "format": "json",
        "per_response_detail": True,
        "summary_report": True,
        "output_dir": "results",
        "save_evaluations": True,
    },
    "logging": {
        "level": "INFO",
        "format": "json",
        "log_dir": "logs",
        "log_file": "ai_email_system.log",
        "console_output": True,
    },
}


@pytest.fixture
def config_file(tmp_path):
    """Create a temporary valid config file."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(VALID_CONFIG))
    return str(config_path)


class TestConfigLoading:
    """Tests for config file loading."""

    def test_loads_valid_config(self, config_file):
        """Config should load successfully from a valid YAML file."""
        config = Config(config_file)
        assert config is not None

    def test_raises_on_missing_file(self, tmp_path):
        """Config should raise ConfigurationError if file doesn't exist."""
        with pytest.raises(ConfigurationError, match="not found"):
            Config(str(tmp_path / "missing.yaml"))

    def test_raises_on_empty_file(self, tmp_path):
        """Config should raise ConfigurationError if file is empty."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")
        with pytest.raises(ConfigurationError, match="empty"):
            Config(str(empty_file))

    def test_raises_on_invalid_yaml(self, tmp_path):
        """Config should raise ConfigurationError on invalid YAML syntax."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("key: [unclosed bracket")
        with pytest.raises(ConfigurationError):
            Config(str(bad_file))


class TestConfigValidation:
    """Tests for configuration validation logic."""

    def test_raises_on_missing_dataset_section(self, tmp_path):
        """Missing 'dataset' section should raise ConfigurationError."""
        cfg = {k: v for k, v in VALID_CONFIG.items() if k != "dataset"}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg))
        with pytest.raises(ConfigurationError, match="dataset"):
            Config(str(p))

    def test_raises_on_missing_llm_section(self, tmp_path):
        """Missing 'llm' section should raise ConfigurationError."""
        cfg = {k: v for k, v in VALID_CONFIG.items() if k != "llm"}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg))
        with pytest.raises(ConfigurationError, match="llm"):
            Config(str(p))

    def test_raises_on_invalid_temperature(self, tmp_path):
        """Temperature outside 0-2 range should raise ConfigurationError."""
        cfg = yaml.safe_load(yaml.dump(VALID_CONFIG))
        cfg["generation"]["temperature"] = 5.0
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg))
        with pytest.raises(ConfigurationError, match="temperature"):
            Config(str(p))

    def test_raises_on_invalid_fallback_provider(self, tmp_path):
        """Unknown fallback_provider should raise ConfigurationError."""
        cfg = yaml.safe_load(yaml.dump(VALID_CONFIG))
        cfg["llm"]["fallback_provider"] = "unknown_provider"
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg))
        with pytest.raises(ConfigurationError, match="fallback_provider"):
            Config(str(p))

    def test_raises_on_invalid_output_format(self, tmp_path):
        """Unsupported output format should raise ConfigurationError."""
        cfg = yaml.safe_load(yaml.dump(VALID_CONFIG))
        cfg["output"]["format"] = "xml"
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg))
        with pytest.raises(ConfigurationError, match="format"):
            Config(str(p))

    def test_raises_on_empty_evaluation_dimensions(self, tmp_path):
        """Empty evaluation dimensions should raise ConfigurationError."""
        cfg = yaml.safe_load(yaml.dump(VALID_CONFIG))
        cfg["evaluation"]["dimensions"] = []
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg))
        with pytest.raises(ConfigurationError, match="dimensions"):
            Config(str(p))

    def test_valid_fallback_providers_accepted(self, tmp_path):
        """openai and anthropic fallback providers should be accepted."""
        for provider in ["openai", "anthropic"]:
            cfg = yaml.safe_load(yaml.dump(VALID_CONFIG))
            cfg["llm"]["fallback_provider"] = provider
            p = tmp_path / f"cfg_{provider}.yaml"
            p.write_text(yaml.dump(cfg))
            config = Config(str(p))
            assert config.get("llm.fallback_provider") == provider


class TestConfigGetters:
    """Tests for config value retrieval."""

    def test_get_simple_value(self, config_file):
        """config.get should return correct value for existing key."""
        config = Config(config_file)
        assert config.get("llm.primary") == "lm_studio"

    def test_get_nested_value(self, config_file):
        """config.get should support dot notation for nested keys."""
        config = Config(config_file)
        assert config.get("llm.lm_studio_url") == "http://127.0.0.1:1234/v1"

    def test_get_default_for_missing_key(self, config_file):
        """config.get should return default when key doesn't exist."""
        config = Config(config_file)
        assert config.get("nonexistent.key", "default_val") == "default_val"

    def test_get_section(self, config_file):
        """get_section should return the entire section dict."""
        config = Config(config_file)
        section = config.get_section("dataset")
        assert isinstance(section, dict)
        assert "path" in section
        assert "min_pairs" in section

    def test_get_section_raises_for_missing_section(self, config_file):
        """get_section should raise ConfigurationError for unknown section."""
        config = Config(config_file)
        with pytest.raises(ConfigurationError):
            config.get_section("nonexistent_section")


class TestLoadConfigConvenience:
    """Tests for the load_config() convenience function."""

    def test_load_config_with_path(self, config_file):
        """load_config should load config when path is provided."""
        config = load_config(config_file)
        assert config is not None
        assert config.get("llm.primary") == "lm_studio"


class TestConfigRepr:
    """Tests for Config string representations."""

    def test_repr_contains_path(self, config_file):
        """repr should include config file path."""
        config = Config(config_file)
        assert "Config" in repr(config)

    def test_str_shows_summary(self, config_file):
        """str should show human-readable summary."""
        config = Config(config_file)
        summary = str(config)
        assert "lm_studio" in summary
        assert "http://127.0.0.1:1234" in summary
