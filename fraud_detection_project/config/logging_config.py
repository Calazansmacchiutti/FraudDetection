"""
Logging Configuration

This module sets up logging for the fraud detection system.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from config.settings import settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the application.
    
    Parameters
    ----------
    log_level : str, optional
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : str, optional
        Path to log file
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Use settings if not provided
    log_level = log_level or settings.log.log_level
    log_file = log_file or str(settings.log.log_dir / settings.log.log_file)
    
    # Create logger
    logger = logging.getLogger("fraud_detection")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatters
    formatter = logging.Formatter(settings.log.log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=settings.log.max_bytes,
        backupCount=settings.log.backup_count
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Parameters
    ----------
    name : str
        Name of the module
        
    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(f"fraud_detection.{name}")


# Initialize default logger
logger = setup_logging()
