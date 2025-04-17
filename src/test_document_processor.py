import os
import logging
from document_processor import DocumentProcessor
from utils import ensure_directory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_single_document():
    """Test processing a single document."""
    ensure_directory("data/raw")

    # Check if there are PDF files in the raw directory
    pdf_files = [f for f in os.listdir("data/raw") if f.lower().endswith('.pdf')]

    if not pdf_files:
        logger.error("No PDF files found in data/raw directory!")
        return

    # Process the first PDF
    test_file = os.path.join("data/raw", pdf_files[0])
    logger.info(f"Testing document processor with file: {test_file}")

    processor = DocumentProcessor()
    result = processor.process_document(test_file)

    if result.get("success", False):
        logger.info(f"Successfully processed {test_file}")
        logger.info(f"Title: {result.get('metadata', {}).get('title', 'Unknown')}")
        logger.info(f"Created {len(result.get('chunks', []))} chunks")

        # Print the first chunk for inspection
        if result.get("chunks"):
            logger.info("First chunk content:")
            logger.info(f"Section: {result['chunks'][0]['section']}")
            logger.info(f"Text sample: {result['chunks'][0]['text'][:200]}...")
    else:
        logger.error(f"Failed to process {test_file}")


def test_processor():
    """Test the document processor on all files."""
    processor = DocumentProcessor()
    results = processor.process_all_documents()

    successful = sum(1 for doc in results if doc.get("success", False))
    logger.info(f"Processed {len(results)} documents, {successful} successfully")

    # Print summary of chunks created
    total_chunks = sum(len(doc.get("chunks", [])) for doc in results if doc.get("success", False))
    logger.info(f"Created a total of {total_chunks} chunks across all documents")


if __name__ == "__main__":
    test_single_document()  # For detailed debug info on a single document
    # test_processor()      # Uncomment to process all documents
