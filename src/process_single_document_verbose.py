# src/process_single_document_verbose.py
import os
import sys
import logging
import argparse
import time
from document_processor import DocumentProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Process a single academic paper with verbose logging")
    parser.add_argument("--file", type=str, required=True, help="Path to PDF file to process")
    parser.add_argument("--chunk-size", type=int, default=800, help="Maximum chunk size in characters")
    parser.add_argument("--overlap", type=int, default=150, help="Overlap between chunks in characters")

    args = parser.parse_args()

    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        return 1

    processor = DocumentProcessor()

    # Step 1: Extract text from PDF
    logger.info(f"STEP 1: Extracting text from {args.file}...")
    start_time = time.time()
    pages = processor.extract_text_from_pdf(args.file)
    if not pages:
        logger.error("Text extraction failed or returned no pages")
        return 1
    logger.info(f"Text extraction completed in {time.time() - start_time:.2f} seconds")

    # Step 2: Extract metadata
    logger.info("STEP 2: Extracting metadata...")
    start_time = time.time()
    metadata = processor.extract_metadata(args.file, pages)
    logger.info(f"Metadata extraction completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Title: {metadata.get('title', 'Unknown')}")

    # Step 3: Identify document structure
    logger.info("STEP 3: Identifying document structure...")
    start_time = time.time()
    document_structure = processor.identify_document_structure(pages)
    sections_count = len(document_structure.get("sections", []))
    logger.info(f"Document structure identification completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Found {sections_count} sections")

    # Log section titles to help with debugging
    for i, section in enumerate(document_structure.get("sections", [])):
        logger.info(f"Section {i + 1}: {section.get('title', 'Untitled')} - {len(section.get('text', ''))} chars")

    # Step 4: Chunk document
    logger.info("STEP 4: Chunking document...")
    start_time = time.time()
    chunks = processor.chunk_document(document_structure.get("sections", []), args.chunk_size, args.overlap)
    logger.info(f"Chunking completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Created {len(chunks)} chunks")

    # Step 5: Save the processed document
    logger.info("STEP 5: Saving processed document...")
    start_time = time.time()

    # Create a processed document
    processed_document = {
        "metadata": metadata,
        "chunks": chunks,
        "success": True
    }

    # Save to JSON
    if metadata.get("title"):
        import re
        from utils import ensure_directory, save_json

        ensure_directory("data/processed")
        safe_title = re.sub(r'[^a-zA-Z0-9]', '_', metadata["title"])[:100]
        output_path = os.path.join("data/processed", f"{safe_title}.json")
        save_json(processed_document, output_path)
        logger.info(f"Saved processed document to {output_path} in {time.time() - start_time:.2f} seconds")

    logger.info("Processing completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
