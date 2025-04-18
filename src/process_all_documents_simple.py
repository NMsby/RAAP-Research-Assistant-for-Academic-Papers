import os
import sys
import logging
import argparse
from tqdm.auto import tqdm
from process_single_document_simple import process_document_simple
from utils import ensure_directory, save_json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Process all academic papers with simplified approach")
    parser.add_argument("--input-dir", type=str, default="data/raw", help="Directory containing PDF files")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to save processed files")
    parser.add_argument("--chunk-size", type=int, default=700, help="Maximum chunk size in characters")
    parser.add_argument("--max-pages", type=int, default=100, help="Maximum pages to process per document")

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1

    ensure_directory(args.output_dir)

    # Get all PDF files
    pdf_files = [f for f in os.listdir(args.input_dir)
                 if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(args.input_dir, f))]

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    results = []
    for pdf_file in tqdm(pdf_files, desc="Processing documents"):
        pdf_path = os.path.join(args.input_dir, pdf_file)
        try:
            success = process_document_simple(
                pdf_path,
                args.output_dir,
                args.chunk_size,
                args.max_pages
            )

            result = {
                "file": pdf_file,
                "success": success
            }
            results.append(result)

        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {str(e)}")
            results.append({
                "file": pdf_file,
                "success": False,
                "error": str(e)
            })

    # Create summary
    summary = {
        "total": len(results),
        "successful": sum(1 for r in results if r.get("success", False)),
        "results": results
    }

    save_json(summary, os.path.join(args.output_dir, "processing_summary_simple.json"))

    logger.info(f"Processing complete. {summary['successful']}/{summary['total']} documents processed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
