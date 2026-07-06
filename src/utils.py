"""
Shared utility functions for the AI Email Response System.

Provides file I/O helpers, logging setup, and common utility functions
used across all system components.
"""

import os
import sys
import json
import logging
import logging.handlers
import time
import functools
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """
    Formats log records as JSON for structured logging.
    Useful for log aggregation and parsing with tools like ELK, CloudWatch, etc.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as a JSON string."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Include exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Include extra fields added to log records
        extra_keys = {
            k: v for k, v in record.__dict__.items()
            if k not in (
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                'thread', 'threadName', 'processName', 'process', 'message',
                'taskName',
            )
        }
        if extra_keys:
            log_entry["extra"] = extra_keys
        
        return json.dumps(log_entry, default=str)


def setup_logging(
    name: str = "ai_email_system",
    level: str = "INFO",
    log_format: str = "json",
    log_dir: str = "logs",
    log_file: Optional[str] = None,
    console_output: bool = True,
) -> logging.Logger:
    """
    Set up structured logging for the system.
    
    Args:
        name: Logger name (use __name__ for module-level loggers)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format ("json" or "text")
        log_dir: Directory to store log files
        log_file: Log file name (default: ai_email_system.log)
        console_output: Whether to also log to console
        
    Returns:
        Configured Logger instance
        
    Example:
        >>> logger = setup_logging(name=__name__)
        >>> logger.info("System started", extra={"component": "main"})
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Create formatters
    if log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # Use text format for console to improve readability
        if log_format == "json":
            console_formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        else:
            console_formatter = formatter
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"ai_email_system.{name}")


# ---------------------------------------------------------------------------
# File I/O Helpers
# ---------------------------------------------------------------------------

def load_json(filepath: Union[str, Path]) -> Any:
    """
    Load JSON data from a file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Parsed JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Save data as JSON to a file.
    
    Creates parent directories if they don't exist.
    
    Args:
        data: Data to serialize as JSON
        filepath: Destination file path
        indent: JSON indentation level
        
    Raises:
        TypeError: If data is not JSON serializable
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, default=str, ensure_ascii=False)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object for the created/existing directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def list_json_files(directory: Union[str, Path]) -> List[Path]:
    """
    List all JSON files in a directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of Path objects for JSON files
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        return []
    
    return sorted(dir_path.glob("*.json"))


def read_text_file(filepath: Union[str, Path]) -> str:
    """
    Read text content from a file.
    
    Args:
        filepath: Path to text file
        
    Returns:
        File content as string
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def write_text_file(content: str, filepath: Union[str, Path]) -> None:
    """
    Write text content to a file.
    
    Creates parent directories if they don't exist.
    
    Args:
        content: Text content to write
        filepath: Destination file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Timing and Performance Utilities
# ---------------------------------------------------------------------------

def timer(func):
    """
    Decorator to measure and log function execution time.
    
    Usage:
        @timer
        def my_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        
        logger = get_logger("timer")
        logger.debug(
            f"{func.__qualname__} completed in {elapsed:.3f}s",
            extra={"function": func.__qualname__, "elapsed_seconds": elapsed}
        )
        
        return result
    return wrapper


class Timer:
    """
    Context manager for timing code blocks.
    
    Usage:
        with Timer("embedding_generation") as t:
            embeddings = model.encode(texts)
        print(f"Elapsed: {t.elapsed:.3f}s")
    """
    
    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed = 0.0
        self._start = None
    
    def __enter__(self):
        self._start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
    
    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed * 1000


# ---------------------------------------------------------------------------
# Text Processing Utilities
# ---------------------------------------------------------------------------

def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Input text
        max_length: Maximum number of characters
        suffix: String to append when truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def sanitize_text(text: str) -> str:
    """
    Remove control characters and normalize whitespace in text.
    
    Args:
        text: Input text
        
    Returns:
        Sanitized text
    """
    # Remove null bytes and other control characters except newlines/tabs
    text = ''.join(c for c in text if c == '\n' or c == '\t' or not c.isspace() or c == ' ')
    
    # Normalize multiple whitespace
    import re
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def count_characters(text: str, include_spaces: bool = True) -> int:
    """Count characters in text."""
    if include_spaces:
        return len(text)
    return len(text.replace(' ', ''))


# ---------------------------------------------------------------------------
# Validation Utilities
# ---------------------------------------------------------------------------

def validate_email_text(text: str, min_length: int = 50, max_length: int = 2000) -> tuple[bool, str]:
    """
    Validate email text length and content.
    
    Args:
        text: Email text to validate
        min_length: Minimum character length
        max_length: Maximum character length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Email text cannot be empty"
    
    text = text.strip()
    
    if len(text) < min_length:
        return False, f"Email text too short: {len(text)} chars (minimum {min_length})"
    
    if len(text) > max_length:
        return False, f"Email text too long: {len(text)} chars (maximum {max_length})"
    
    return True, ""


# ---------------------------------------------------------------------------
# Result Formatting
# ---------------------------------------------------------------------------

def format_score(score: float, scale: float = 1.0, precision: int = 3) -> float:
    """
    Format a numeric score to consistent precision.
    
    Args:
        score: Raw score
        scale: Scale factor (1.0 for 0-1 scale, 5.0 for 1-5 scale)
        precision: Decimal places
        
    Returns:
        Formatted score
    """
    return round(float(score), precision)


def generate_id(prefix: str = "item", index: Optional[int] = None) -> str:
    """
    Generate a unique identifier.
    
    Args:
        prefix: ID prefix
        index: Optional sequential index
        
    Returns:
        Generated ID string
        
    Example:
        >>> generate_id("email", 1)
        "email_001"
        >>> generate_id("eval")
        "eval_1704067200"
    """
    if index is not None:
        return f"{prefix}_{index:03d}"
    
    timestamp = int(time.time())
    return f"{prefix}_{timestamp}"
