import os
import sys
import logging
import argparse
from document_processor import DocumentProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Process a single academic paper")
    parser.add_argument("--file", type=str, required=True, help="Path to PDF file to process")
    parser.add_argument("--chunk-size", type=int, default=800, help="Maximum chunk size in characters")
    parser.add_argument("--overlap", type=int, default=150, help="Overlap between chunks in characters")

    args = parser.parse_args()

    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        return 1

    processor = DocumentProcessor()
    result = processor.process_document(args.file, args.chunk_size, args.overlap)

    if result.get("success", False):
        logger.info(f"Successfully processed {args.file}")
        logger.info(f"Title: {result.get('metadata', {}).get('title', 'Unknown')}")
        logger.info(f"Created {len(result.get('chunks', []))} chunks")
        return 0
    else:
        logger.error(f"Failed to process {args.file}: {result.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
