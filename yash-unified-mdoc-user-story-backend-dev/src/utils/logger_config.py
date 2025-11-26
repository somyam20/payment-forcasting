# logger_config.py
import logging
from logging.handlers import TimedRotatingFileHandler

def setup_logger():
    handler = TimedRotatingFileHandler('app.log', when='midnight', interval=1)
    handler.suffix = "%Y-%m-%d"
    logging.basicConfig(
        handlers=[handler],
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def setup_usage_logger():
    logger = logging.getLogger("usageLogger")
    if not logger.handlers:  # ✅ Prevent duplicate handlers
        logger.setLevel(logging.INFO)
        handler = TimedRotatingFileHandler("usage.log", when="midnight", interval=1)
        handler.suffix = "%Y-%m-%d"
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger
