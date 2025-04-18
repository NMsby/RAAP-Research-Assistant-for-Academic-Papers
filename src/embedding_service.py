import os
import logging
import time
from typing import List, Dict, Any, Optional, Union
from tqdm.auto import tqdm
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

from utils import ensure_directory, save_json, load_json

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and managing text embeddings."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "models/text-embedding-004"):
        """Initialize the embedding service.

        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env variable)
            model_name: Name of the embedding model to use
        """
        # Configure API
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("No API key provided. Set GOOGLE_API_KEY environment variable.")

        genai.configure(api_key=api_key)
        self.model_name = model_name
        logger.info(f"Embedding service initialized with model: {model_name}")

        # Check if embeddings directory exists
        self.embeddings_dir = "data/embeddings"
        ensure_directory(self.embeddings_dir)

    def generate_embedding(self, text: str,
                           task_type: str = "retrieval_document",
                           retry_count: int = 3,
                           retry_delay: float = 2.0) -> List[float]:
        """Generate embedding for a text string.

        Args:
            text: Text to embed
            task_type: Type of embedding task (retrieval_document, retrieval_query, etc.)
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds

        Returns:
            Embedding vector as a list of floats
        """
        attempts = 0
        while attempts < retry_count:
            try:
                response = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type=task_type
                )

                # Extract and return the embedding vector
                return response["embedding"]

            except Exception as e:
                attempts += 1
                logger.warning(f"Embedding generation failed (attempt {attempts}/{retry_count}): {str(e)}")
                if attempts < retry_count:
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to generate embedding after {retry_count} attempts")
                    raise

    def batch_generate_embeddings(self, texts: List[str],
                                  task_type: str = "retrieval_document",
                                  batch_size: int = 5) -> List[List[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed
            task_type: Type of embedding task
            batch_size: Number of embeddings to generate in each batch

        Returns:
            List of embedding vectors
        """
        embeddings = []

        # Process in batches to avoid rate limits
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []

            for text in batch_texts:
                embedding = self.generate_embedding(text, task_type)
                batch_embeddings.append(embedding)

            embeddings.extend(batch_embeddings)

            # Sleep to avoid rate limits (if not the last batch)
            if i + batch_size < len(texts):
                time.sleep(0.5)

        return embeddings

    def create_document_embeddings(self, document_path: str,
                                   output_name: Optional[str] = None,
                                   task_type: str = "retrieval_document") -> Dict[str, Any]:
        """Create embeddings for all chunks in a processed document.

        Args:
            document_path: Path to processed document JSON file
            output_name: Name for the output embeddings file (default to document name)
            task_type: Type of embedding task

        Returns:
            Dictionary with document metadata and embeddings
        """
        # Load the processed document
        try:
            document = load_json(document_path)
        except Exception as e:
            logger.error(f"Failed to load document {document_path}: {str(e)}")
            return {"success": False, "error": str(e)}

        if not document.get("chunks"):
            logger.error(f"No chunks found in document {document_path}")
            return {"success": False, "error": "No chunks found"}

        # Extract the chunks and generate embeddings
        chunks = document["chunks"]
        texts = [chunk["text"] for chunk in chunks]

        logger.info(f"Generating embeddings for {len(texts)} chunks from {document_path}")
        try:
            embeddings = self.batch_generate_embeddings(texts, task_type)
        except Exception as e:
            logger.error(f"Failed to generate embeddings for {document_path}: {str(e)}")
            return {"success": False, "error": str(e)}

        # Create result dictionary
        result = {
            "metadata": document.get("metadata", {}),
            "chunks": [
                {
                    **chunks[i],
                    "embedding": embeddings[i]
                }
                for i in range(len(chunks))
            ],
            "success": True
        }

        # Save to file
        if output_name is None:
            # Use the document title or filename
            title = document.get("metadata", {}).get("title", "")
            if not title:
                title = os.path.splitext(os.path.basename(document_path))[0]

            output_name = f"{title.replace(' ', '_')[:100]}_embeddings.json"

        output_path = os.path.join(self.embeddings_dir, output_name)
        save_json(result, output_path)
        logger.info(f"Saved embeddings to {output_path}")

        return result

    def create_all_document_embeddings(self, documents_dir: str = "data/processed",
                                       task_type: str = "retrieval_document") -> Dict[str, Any]:
        """Create embeddings for all processed documents.

        Args:
            documents_dir: Directory containing processed documents
            task_type: Type of embedding task

        Returns:
            Dictionary with results' summary
        """
        # Get all processed document files
        if not os.path.exists(documents_dir):
            logger.error(f"Documents directory not found: {documents_dir}")
            return {"success": False, "error": "Directory not found"}

        json_files = [f for f in os.listdir(documents_dir)
                      if f.endswith(".json") and not f.startswith("processing_summary")]

        if not json_files:
            logger.error(f"No processed documents found in {documents_dir}")
            return {"success": False, "error": "No documents found"}

        logger.info(f"Processing embeddings for {len(json_files)} documents")

        results = []
        for json_file in tqdm(json_files, desc="Creating document embeddings"):
            document_path = os.path.join(documents_dir, json_file)
            result = self.create_document_embeddings(document_path, task_type=task_type)

            results.append({
                "file": json_file,
                "success": result.get("success", False),
                "error": result.get("error", None)
            })

        # Create summary
        summary = {
            "total": len(results),
            "successful": sum(1 for r in results if r.get("success", False)),
            "results": results
        }

        summary_path = os.path.join(self.embeddings_dir, "embedding_summary.json")
        save_json(summary, summary_path)

        return summary


if __name__ == "__main__":
    # Example usage
    embedding_service = EmbeddingService()

    # Create embeddings for all processed documents
    result = embedding_service.create_all_document_embeddings()

    logger.info(
        f"Embedding generation complete. {result['successful']}/{result['total']} documents processed successfully.")
