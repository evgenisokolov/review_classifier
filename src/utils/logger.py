import logging
from pythonjsonlogger import jsonlogger

def init_logger(name: str = "review_classifier", level=logging.INFO) -> logging.Logger:
    """Initialize and return a structured JSON logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger