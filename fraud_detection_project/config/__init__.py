"""
Configuration Module

This module provides configuration settings for the fraud detection system.
"""

from config.settings import settings, Settings
from config.logging_config import setup_logging, get_logger, logger

__all__ = [
    "settings",
    "Settings",
    "setup_logging",
    "get_logger",
    "logger"
]
