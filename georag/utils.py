import re
import logging

def setup_logger(name="GeoRAG"):
    """Sets up a console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def clean_text(text: str) -> str:
    """Cleans text by removing excessive whitespace and artifacts."""
    if not text:
        return ""
    # Replace multiple newlines with single space
    text = re.sub(r'\n+', ' ', text)
    # Remove excessive spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
