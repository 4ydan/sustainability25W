"""Logging configuration for the application."""

import logging
import os
import sys
import colorlog


def setup_logger(name: str = __name__) -> logging.Logger:
    """
    Set up logger with appropriate level based on DEBUG environment variable.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        # Determine log level from environment
        debug_enabled = os.getenv("DEBUG", "0") == "1"
        log_level = logging.DEBUG if debug_enabled else logging.INFO

        # Create console handler
        handler = colorlog.StreamHandler(sys.stdout)
        handler.setLevel(log_level)

        # Create colored formatter
        formatter = colorlog.ColoredFormatter(
            fmt="%(log_color)s%(levelname)s%(reset)s: %(message)s",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            reset=True,
            style='%'
        )
        handler.setFormatter(formatter)

        # Configure logger
        logger.addHandler(handler)
        logger.setLevel(log_level)
        logger.propagate = False

    return logger
