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
        # Only color the level name
        color = getattr(LogColors, record.levelname, LogColors.RESET)
        colored_level = f"{color}%(levelname)s{LogColors.RESET}"

        # Build the format string with only the level colored
        log_fmt = f"%(asctime)s [{colored_level}] %(message)s"

        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter())
        logger.addHandler(handler)

    return logger
