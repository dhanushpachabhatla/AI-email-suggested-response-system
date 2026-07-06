"""
Tests for shared utility functions (src/utils.py).

Validates:
- Logging setup and JSON formatter
- File I/O helper functions
- Text processing utilities
- Timer utilities
- Validation helpers
"""

import json
import logging
import sys
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    setup_logging,
    JSONFormatter,
    get_logger,
    load_json,
    save_json,
    ensure_directory,
    list_json_files,
    read_text_file,
    write_text_file,
    Timer,
    truncate_text,
    sanitize_text,
    count_words,
    count_characters,
    validate_email_text,
    format_score,
    generate_id,
)


class TestJSONFormatter:
    """Tests for JSON log formatter."""

    def test_produces_valid_json(self):
        """JSONFormatter should produce parseable JSON for each log record."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "Test message"
        assert parsed["level"] == "INFO"

    def test_includes_required_fields(self):
        """JSON log records should include timestamp, level, logger, and message."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="mylogger", level=logging.WARNING,
            pathname="module.py", lineno=42,
            msg="Warning!", args=(), exc_info=None
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "timestamp" in parsed
        assert "level" in parsed
        assert "logger" in parsed
        assert "message" in parsed


class TestSetupLogging:
    """Tests for logging setup."""

    def test_returns_logger(self, tmp_path):
        """setup_logging should return a Logger instance."""
        log_file = "test.log"
        logger = setup_logging(
            name="test_setup_logging_unique",
            level="DEBUG",
            log_format="json",
            log_dir=str(tmp_path),
            log_file=log_file,
        )
        assert isinstance(logger, logging.Logger)

    def test_creates_log_file(self, tmp_path):
        """setup_logging should create the specified log file."""
        logger = setup_logging(
            name="test_creates_file_unique",
            level="INFO",
            log_format="json",
            log_dir=str(tmp_path),
            log_file="output.log",
        )
        logger.info("Test entry")
        
        # Flush handlers
        for handler in logger.handlers:
            handler.flush()
        
        assert (tmp_path / "output.log").exists()

    def test_no_duplicate_handlers(self, tmp_path):
        """Calling setup_logging twice on the same logger should not add duplicate handlers."""
        name = "no_duplicate_test_unique"
        logger1 = setup_logging(name=name, log_dir=str(tmp_path), log_file="dup.log")
        initial_count = len(logger1.handlers)
        logger2 = setup_logging(name=name, log_dir=str(tmp_path), log_file="dup.log")
        assert len(logger2.handlers) == initial_count


class TestFileIOHelpers:
    """Tests for file I/O helper functions."""

    def test_save_and_load_json(self, tmp_path):
        """save_json + load_json round-trip should preserve data."""
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        filepath = tmp_path / "test.json"
        save_json(data, str(filepath))
        loaded = load_json(str(filepath))
        assert loaded == data

    def test_load_json_raises_on_missing_file(self, tmp_path):
        """load_json should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_json(str(tmp_path / "nonexistent.json"))

    def test_save_json_creates_directories(self, tmp_path):
        """save_json should create nested directories if they don't exist."""
        filepath = tmp_path / "nested" / "deep" / "file.json"
        save_json({"key": "val"}, str(filepath))
        assert filepath.exists()

    def test_ensure_directory_creates_dir(self, tmp_path):
        """ensure_directory should create the directory."""
        new_dir = tmp_path / "new_dir"
        result = ensure_directory(str(new_dir))
        assert new_dir.exists()
        assert result == new_dir

    def test_ensure_directory_is_idempotent(self, tmp_path):
        """ensure_directory should not fail if directory already exists."""
        existing = tmp_path / "existing"
        existing.mkdir()
        result = ensure_directory(str(existing))  # Should not raise
        assert result == existing

    def test_list_json_files(self, tmp_path):
        """list_json_files should return all .json files in a directory."""
        (tmp_path / "a.json").write_text("{}")
        (tmp_path / "b.json").write_text("{}")
        (tmp_path / "c.txt").write_text("text")
        files = list_json_files(str(tmp_path))
        assert len(files) == 2
        assert all(f.suffix == ".json" for f in files)

    def test_list_json_files_empty_dir(self, tmp_path):
        """list_json_files should return empty list for directory with no JSON files."""
        files = list_json_files(str(tmp_path))
        assert files == []

    def test_list_json_files_missing_dir(self, tmp_path):
        """list_json_files should return empty list for non-existent directory."""
        files = list_json_files(str(tmp_path / "does_not_exist"))
        assert files == []

    def test_write_and_read_text_file(self, tmp_path):
        """write_text_file + read_text_file round-trip should preserve content."""
        content = "Hello, World!\nLine 2"
        filepath = tmp_path / "test.txt"
        write_text_file(content, str(filepath))
        assert read_text_file(str(filepath)) == content

    def test_read_text_file_raises_on_missing(self, tmp_path):
        """read_text_file should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            read_text_file(str(tmp_path / "missing.txt"))


class TestTimerUtility:
    """Tests for Timer context manager."""

    def test_timer_measures_elapsed(self):
        """Timer should record elapsed time after context exit."""
        import time
        with Timer("test") as t:
            time.sleep(0.01)
        assert t.elapsed > 0

    def test_timer_elapsed_ms(self):
        """Timer.elapsed_ms should return milliseconds."""
        import time
        with Timer() as t:
            time.sleep(0.01)
        assert t.elapsed_ms >= 10  # at least 10ms


class TestTextProcessing:
    """Tests for text utility functions."""

    def test_truncate_short_text(self):
        """truncate_text should not modify text shorter than max_length."""
        text = "Hello"
        assert truncate_text(text, max_length=100) == "Hello"

    def test_truncate_long_text(self):
        """truncate_text should trim text exceeding max_length."""
        text = "A" * 200
        result = truncate_text(text, max_length=50)
        assert len(result) == 50
        assert result.endswith("...")

    def test_sanitize_text_strips_whitespace(self):
        """sanitize_text should strip leading/trailing whitespace."""
        assert sanitize_text("  hello  ") == "hello"

    def test_count_words(self):
        """count_words should return correct word count."""
        assert count_words("hello world foo") == 3
        assert count_words("one") == 1

    def test_count_characters_with_spaces(self):
        """count_characters should include spaces by default."""
        assert count_characters("hello world") == 11

    def test_count_characters_without_spaces(self):
        """count_characters should exclude spaces when include_spaces=False."""
        assert count_characters("hello world", include_spaces=False) == 10


class TestValidateEmailText:
    """Tests for email text validation."""

    def test_valid_email_passes(self):
        """Email text within length bounds should pass validation."""
        text = "A" * 100
        is_valid, msg = validate_email_text(text)
        assert is_valid is True
        assert msg == ""

    def test_empty_text_fails(self):
        """Empty email text should fail validation."""
        is_valid, msg = validate_email_text("")
        assert is_valid is False
        assert "empty" in msg.lower()

    def test_text_too_short_fails(self):
        """Text shorter than min_length should fail validation."""
        is_valid, msg = validate_email_text("Hi", min_length=50)
        assert is_valid is False
        assert "short" in msg.lower()

    def test_text_too_long_fails(self):
        """Text longer than max_length should fail validation."""
        is_valid, msg = validate_email_text("A" * 3000, max_length=2000)
        assert is_valid is False
        assert "long" in msg.lower()

    def test_whitespace_only_fails(self):
        """Whitespace-only text should fail validation."""
        is_valid, msg = validate_email_text("   \n\t  ")
        assert is_valid is False


class TestFormatAndGenerateId:
    """Tests for score formatting and ID generation."""

    def test_format_score_rounds_correctly(self):
        """format_score should round to specified precision."""
        assert format_score(0.12345, precision=3) == 0.123

    def test_generate_id_with_index(self):
        """generate_id with index should produce zero-padded string."""
        assert generate_id("email", 1) == "email_001"
        assert generate_id("email", 42) == "email_042"

    def test_generate_id_without_index(self):
        """generate_id without index should produce timestamp-based string."""
        result = generate_id("eval")
        assert result.startswith("eval_")
        assert len(result) > 5
