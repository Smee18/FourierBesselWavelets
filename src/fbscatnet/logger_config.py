import logging
import sys


class LogColors:
    RESET = "\033[0m"
    DEBUG = "\033[36m"  # Cyan
    INFO = "\033[32m"  # Green
    WARNING = "\033[33m"  # Yellow
    ERROR = "\033[31m"  # Red
    CRITICAL = "\033[35m"  # Magenta


class ColoredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = getattr(LogColors, record.levelname, LogColors.RESET)
        colored_level = f"{color}%(levelname)s{LogColors.RESET}"
        log_fmt = f"%(asctime)s [{colored_level}] %(message)s"
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_logger(name: str) -> logging.Logger:
    """Internal function to set up the library logger with a NullHandler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def enable_colored_logs(level: int = logging.INFO) -> None:
    """
    Helper function for users to easily enable colored console logging for fbscatnet.

    Usage:
        import fbscatnet
        fbscatnet.enable_colored_logs()
    """
    # Get the top-level logger for the package
    logger = logging.getLogger("fbscatnet")
    logger.setLevel(level)

    # Check if we already added a StreamHandler to avoid duplicates
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter())
        logger.addHandler(handler)
