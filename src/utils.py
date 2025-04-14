import os
import logging
from typing import Dict, List, Any, Tuple, Optional
from tqdm.auto import tqdm
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_directory(directory_path: str) -> None:
    """Ensure a directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")

def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved data to {filepath}")

def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded data from {filepath}")
    return data