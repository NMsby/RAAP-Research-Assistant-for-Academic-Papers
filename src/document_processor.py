import os
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
import pdfplumber
from tqdm.auto import tqdm

from utils import ensure_directory, save_json, load_json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process academic papers for the research assistant."""

    def __init__(self, input_dir: str = "data/raw", output_dir: str = "data/processed"):
        """Initialize the document processor.

        Args:
            input_dir: Directory containing raw PDF files
            output_dir: Directory to store processed documents
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        ensure_directory(output_dir)

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from a PDF file with page information.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of dictionaries with page number and text
        """
        logger.info(f"Extracting text from: {pdf_path}")
        pages = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(tqdm(pdf.pages, desc="Processing pages")):
                    text = page.extract_text() or ""

                    # Some PDFs may have multiple columns - try to handle this
                    if text and len(text.strip()) > 0:
                        pages.append({
                            "page_num": i + 1,
                            "text": text
                        })
                    else:
                        logger.warning(f"Page {i + 1} has no extractable text")

            logger.info(f"Successfully extracted {len(pages)} pages from {pdf_path}")
            return pages

        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return []

    def identify_document_structure(self, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify the structure of a document based on extracted text.

        Args:
            pages: List of page dictionaries with text

        Returns:
            Dictionary with identified sections
        """
        if not pages:
            return {"sections": []}

        # Combine all text to help identify the overall structure
        full_text = "\n\n".join([p["text"] for p in pages])

        # Basic structure identification
        sections = []
        current_section = None
        current_text = []

        # Common section headers in academic papers
        section_patterns = [
            r'^Abstract',
            r'^1\.?\s+Introduction',
            r'^2\.?\s+',
            r'^3\.?\s+',
            r'^4\.?\s+',
            r'^5\.?\s+',
            r'^6\.?\s+',
            r'^7\.?\s+',
            r'^8\.?\s+',
            r'^9\.?\s+',
            r'^10\.?\s+',
            r'^Background',
            r'^Related Work',
            r'^Methodology',
            r'^Method',
            r'^Experiments?',
            r'^Evaluation',
            r'^Results',
            r'^Discussion',
            r'^Conclusion',
            r'^References',
            r'^Appendix',
        ]

        # Combine patterns into a single regex
        section_pattern = '|'.join([f'({p})' for p in section_patterns])

        # Try to identify abstract separately (it's usually at the beginning)
        abstract_match = re.search(r'Abstract[\s\n]+(.*?)(?=Introduction|\d+\.?\s+Introduction|Related Work)',
                                   full_text, re.DOTALL | re.IGNORECASE)

        if abstract_match:
            sections.append({
                "title": "Abstract",
                "text": abstract_match.group(1).strip()
            })

        # Process each page looking for section headers
        for page in pages:
            lines = page["text"].split('\n')

            for line in lines:
                # Check if this line is a section header
                if re.match(section_pattern, line.strip(), re.IGNORECASE):
                    # Save the previous section if it exists
                    if current_section:
                        sections.append({
                            "title": current_section,
                            "text": '\n'.join(current_text).strip()
                        })

                    # Start a new section
                    current_section = line.strip()
                    current_text = []
                else:
                    # Add line to the current section
                    if current_section:
                        current_text.append(line)

        # Add the last section
        if current_section and current_text:
            sections.append({
                "title": current_section,
                "text": '\n'.join(current_text).strip()
            })

        # If no sections found, treat the entire document as one section
        if not sections:
            sections.append({
                "title": "Full Document",
                "text": full_text
            })

        return {"sections": sections}

    def extract_metadata(self, file_path: str, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract metadata from the document.

        Args:
            file_path: Path to the original file
            pages: Extracted pages with text

        Returns:
            Dictionary with metadata
        """
        # Try to find a metadata JSON file first (from paper_collector.py)
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        metadata_path = os.path.join(self.input_dir, f"{base_filename}_metadata.json")

        metadata = {}

        # If a metadata file exists, load it
        if os.path.exists(metadata_path):
            try:
                metadata = load_json(metadata_path)
                logger.info(f"Loaded metadata from {metadata_path}")
            except Exception as e:
                logger.warning(f"Error loading metadata from {metadata_path}: {str(e)}")

        # If no metadata file, try to extract from text
        if not metadata and pages:
            full_text = pages[0]["text"]  # The First page usually has title/authors

            # Try to extract title (usually first line or after header)
            title_match = re.search(r'^(?:\s*arXiv:[\d\.v]+\s*)?([^\n]+)', full_text)
            if title_match:
                metadata["title"] = title_match.group(1).strip()

            # Try to extract authors (usually after title)
            authors_match = re.search(r'\n(.+?)\n+(?:Abstract|Introduction|1\.)', full_text, re.DOTALL)
            if authors_match:
                # Simple heuristic for author names - split by commas, 'and', or newlines
                author_text = authors_match.group(1).strip()
                author_list = re.split(r',|\sand\s|\n', author_text)
                metadata["authors"] = [a.strip() for a in author_list if a.strip()]

        # Add the filename as a fallback title if no title found
        if "title" not in metadata:
            metadata["title"] = base_filename

        # Add the file path
        metadata["file_path"] = file_path

        return metadata

    def chunk_document(self, sections: List[Dict[str, Any]],
                       max_chunk_size: int = 1000,
                       overlap: int = 200) -> List[Dict[str, Any]]:
        """Split document sections into chunks of the appropriate size.

        Args:
            sections: Document sections
            max_chunk_size: Maximum size of a chunk in characters
            overlap: Overlap between chunks in characters

        Returns:
            List of chunk dictionaries
        """
        chunks = []

        for section in sections:
            section_title = section["title"]
            section_text = section["text"]

            # If a section is smaller than max_chunk_size, keep it as one chunk
            if len(section_text) <= max_chunk_size:
                chunks.append({
                    "section": section_title,
                    "text": section_text
                })
                continue

            # Otherwise, split into overlapping chunks
            start = 0
            while start < len(section_text):
                # Calculate end position
                end = min(start + max_chunk_size, len(section_text))

                # If this is not the last chunk, try to break at a sentence or paragraph
                if end < len(section_text):
                    # Look for the last period, newline, or space in the overlap region
                    last_period = section_text.rfind(".", end - 150, end)
                    last_newline = section_text.rfind("\n", end - 150, end)
                    last_space = section_text.rfind(" ", end - 20, end)

                    # Choose the latest breakpoint that exists
                    breakpoint = max(last_period, last_newline, last_space)
                    if breakpoint != -1:
                        end = breakpoint + 1  # Include the breakpoint character

                # Create the chunk
                chunk_text = section_text[start:end]
                chunks.append({
                    "section": section_title,
                    "text": chunk_text,
                    "is_partial": len(section_text) > max_chunk_size
                })

                # Move to the next chunk, with overlap
                start = max(0, end - overlap)

        logger.info(f"Created {len(chunks)} chunks from {len(sections)} sections")
        return chunks

    def process_document(self, file_path: str, max_chunk_size: int = 1000, overlap: int = 200) -> Dict[str, Any]:
        """Process a single document from PDF to structured chunks.

        Args:
            file_path: Path to the PDF file
            max_chunk_size: Maximum size of a chunk in characters
            overlap: Overlap between chunks in characters

        Returns:
            Dictionary with document data and chunks
        """
        # Extract text from PDF
        pages = self.extract_text_from_pdf(file_path)
        if not pages:
            logger.error(f"Failed to extract text from {file_path}")
            return {"success": False, "file_path": file_path}

        # Extract metadata
        metadata = self.extract_metadata(file_path, pages)

        # Identify document structure
        document_structure = self.identify_document_structure(pages)

        # Create chunks
        chunks = self.chunk_document(document_structure["sections"], max_chunk_size, overlap)

        # Combine into a processed document
        processed_document = {
            "metadata": metadata,
            "chunks": chunks,
            "success": True
        }

        # Save processed document
        if metadata.get("title"):
            safe_title = re.sub(r'[^a-zA-Z0-9]', '_', metadata["title"])[:100]
            output_path = os.path.join(self.output_dir, f"{safe_title}.json")
            save_json(processed_document, output_path)
            logger.info(f"Saved processed document to {output_path}")

        return processed_document

    def process_all_documents(self, max_chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Process all PDF documents in the input directory.

        Args:
            max_chunk_size: Maximum size of a chunk in characters
            overlap: Overlap between chunks in characters

        Returns:
            List of processed documents
        """
        logger.info(f"Processing all documents in {self.input_dir}")
        processed_documents = []

        pdf_files = [f for f in os.listdir(self.input_dir)
                     if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(self.input_dir, f))]

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        for pdf_file in tqdm(pdf_files, desc="Processing documents"):
            file_path = os.path.join(self.input_dir, pdf_file)
            processed_document = self.process_document(file_path, max_chunk_size, overlap)
            processed_documents.append(processed_document)

        # Create a summary file
        summary = {
            "total_documents": len(processed_documents),
            "successful_documents": sum(1 for doc in processed_documents if doc.get("success", False)),
            "processed_files": [doc.get("metadata", {}).get("title", "Unknown")
                                for doc in processed_documents if doc.get("success", False)]
        }
        save_json(summary, os.path.join(self.output_dir, "processing_summary.json"))

        return processed_documents


if __name__ == "__main__":
    processor = DocumentProcessor()
    processed_docs = processor.process_all_documents()
    logger.info(f"Processed {len(processed_docs)} documents. "
                f"{sum(1 for doc in processed_docs if doc.get('success', False))} were successful.")
