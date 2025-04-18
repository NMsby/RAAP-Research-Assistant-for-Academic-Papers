# src/process_single_document_simple.py
import os
import sys
import logging
import argparse
import pdfplumber
import re
import json
from typing import List, Dict, Any

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


def process_document_simple(pdf_path: str, output_dir: str = "data/processed",
                            max_chunk_size: int = 800, max_pages: int = 100) -> bool:
    """Process a document with minimal memory usage.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save processed file
        max_chunk_size: Maximum size of text chunks in characters
        max_pages: Maximum number of pages to process

    Returns:
        Success status (True/False)
    """
    # Create output directory
    ensure_directory(output_dir)

    # Get filename
    file_name = os.path.basename(pdf_path)
    file_base = os.path.splitext(file_name)[0]
    output_file = os.path.join(output_dir, f"{file_base}_simple.json")

    metadata = {}
    chunks = []

    # Try to load metadata if it exists
    metadata_path = os.path.splitext(pdf_path)[0] + "_metadata.json"
    if os.path.exists(metadata_path):
        try:
            metadata = load_json(metadata_path)
        except Exception as e:
            logger.warning(f"Error loading metadata: {str(e)}")
            metadata = {"title": file_base}
    else:
        metadata = {"title": file_base}

    # Add the file path to metadata
    metadata["file_path"] = pdf_path

    try:
        # Open PDF
        logger.info(f"Opening PDF: {pdf_path}")
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = min(len(pdf.pages), max_pages)
            logger.info(f"Processing {total_pages} pages out of {len(pdf.pages)} total pages")

            # Process each page individually to minimize memory usage
            for i in range(total_pages):
                logger.info(f"Processing page {i + 1}/{total_pages}")

                try:
                    # Extract text from page
                    page = pdf.pages[i]
                    text = page.extract_text() or ""

                    if text and len(text.strip()) > 0:
                        # Chunk the page text
                        start = 0
                        while start < len(text):
                            end = min(start + max_chunk_size, len(text))

                            # Try to find a good breakpoint if not at the end
                            if end < len(text):
                                for breakpoint_char in ['.', '\n', ' ']:
                                    breakpoint = text.rfind(breakpoint_char, max(0, end - 100), end)
                                    if breakpoint != -1:
                                        end = breakpoint + 1
                                        break

                            chunk_text = text[start:end]
                            chunks.append({
                                "page": i + 1,
                                "section": f"Page {i + 1}",
                                "text": chunk_text
                            })

                            start = end
                except Exception as e:
                    logger.warning(f"Error processing page {i + 1}: {str(e)}")
                    continue

        # Save result
        result = {
            "metadata": metadata,
            "chunks": chunks,
            "success": True
        }

        save_json(result, output_file)
        logger.info(f"Successfully processed {pdf_path} with {len(chunks)} chunks")
        return True

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Process a single academic paper with minimal memory usage")
    parser.add_argument("--file", type=str, required=True, help="Path to PDF file to process")
    parser.add_argument("--chunk-size", type=int, default=800, help="Maximum chunk size in characters")
    parser.add_argument("--max-pages", type=int, default=100, help="Maximum number of pages to process")

    args = parser.parse_args()

    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        return 1

    success = process_document_simple(args.file, max_chunk_size=args.chunk_size, max_pages=args.max_pages)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
