import os
import logging
import shutil
from utils import ensure_directory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def reset_vector_db():
    """Reset the vector database by deleting and recreating its directory."""
    vector_db_path = "data/vector_db"

    if os.path.exists(vector_db_path):
        logger.info(f"Removing existing vector database at {vector_db_path}")
        shutil.rmtree(vector_db_path)

    ensure_directory(vector_db_path)
    logger.info(f"Created fresh vector database directory at {vector_db_path}")

    return True


if __name__ == "__main__":
    reset_vector_db()
    logger.info("Vector database reset complete. Ready for new data.")
