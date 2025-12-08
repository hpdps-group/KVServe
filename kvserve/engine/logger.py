"""
Simple logging utility for kvserve
Controls debug output verbosity
"""

import os
from enum import Enum
from typing import Optional


class LogLevel(Enum):
    """Log level enumeration"""
    ERROR = 0
    WARNING = 1
    INFO = 2
    DEBUG = 3


# Global log level (can be set via environment variable)
_default_log_level = os.environ.get("KVSERVE_LOG_LEVEL", "WARNING").upper()
_global_log_level = LogLevel[_default_log_level] if _default_log_level in LogLevel.__members__ else LogLevel.WARNING


def set_log_level(level: LogLevel):
    """Set global log level"""
    global _global_log_level
    _global_log_level = level


def get_log_level() -> LogLevel:
    """Get current log level"""
    return _global_log_level


def log_debug(message: str):
    """Log debug message"""
    if _global_log_level.value >= LogLevel.DEBUG.value:
        print(f"[DEBUG] {message}")


def log_info(message: str):
    """Log info message"""
    if _global_log_level.value >= LogLevel.INFO.value:
        print(f"[INFO] {message}")


def log_warning(message: str):
    """Log warning message"""
    if _global_log_level.value >= LogLevel.WARNING.value:
        print(f"[WARNING] {message}")


def log_error(message: str):
    """Log error message (always printed)"""
    print(f"[ERROR] {message}")


