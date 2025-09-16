from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
import logging


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger with Rich formatting."""
    logger = logging.getLogger(name if name else __name__)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        console = Console(force_terminal=True)
        handler = RichHandler(console=console, show_time=True, show_level=True, show_path=False)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger




