# src/vector_store.py
import os
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Union
import chromadb
from chromadb.utils import embedding_functions
from tqdm.auto import tqdm
import numpy as np

from utils import ensure_directory, save_json, load_json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VectorStore:
    """Vector database for storing and querying document embeddings."""

    def __init__(self, collection_name: str = "research_papers", persist_directory: str = "data/vector_db"):
        """Initialize the vector store.

        Args:
            collection_name: Name of the vector collection
            persist_directory: Directory to store vector database files
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Create directory if it doesn't exist
        ensure_directory(persist_directory)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Get or create the collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except ValueError:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(name=collection_name)
            logger.info(f"Created new collection: {collection_name}")

    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """Add multiple documents to the vector store.

        Args:
            documents: List of document dictionaries with embeddings
            batch_size: Number of documents to add in each batch
        """
        total_chunks = sum(len(doc.get("chunks", [])) for doc in documents)
        logger.info(f"Adding {total_chunks} chunks from {len(documents)} documents to vector store")

        # Process all documents
        for document in tqdm(documents, desc="Adding documents to vector store"):
            self.add_document(document, batch_size)

    def add_document(self, document: Dict[str, Any], batch_size: int = 100) -> None:
        """Add a document to the vector store.

        Args:
            document: Document dictionary with metadata and embedded chunks
            batch_size: Number of chunks to add in each batch
        """
        # Extract document metadata
        metadata = document.get("metadata", {})
        title = metadata.get("title", "Untitled Document")
        doc_id = metadata.get("id", title.replace(" ", "_")[:40])

        # Process chunks in batches
        chunks = document.get("chunks", [])
        logger.info(f"Adding {len(chunks)} chunks from document: {title}")

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # Prepare data for the batch
            ids = []
            embeddings = []
            documents = []
            metadatas = []

            for j, chunk in enumerate(batch):
                # Create a unique ID for this chunk
                chunk_id = f"{doc_id}_chunk_{i + j}"

                # Extract embedding
                embedding = chunk.get("embedding")
                if not embedding:
                    logger.warning(f"No embedding found for chunk {j} in document {title}")
                    continue

                # Extract text and metadata
                text = chunk.get("text", "")
                chunk_metadata = {
                    "document_id": doc_id,
                    "document_title": title,
                    "section": chunk.get("section", ""),
                    "page": chunk.get("page", 0),
                    **{k: v for k, v in metadata.items() if k not in ["id", "title"]}
                }

                # Add to batch
                ids.append(chunk_id)
                embeddings.append(embedding)
                documents.append(text)
                metadatas.append(chunk_metadata)

            # Add batch to the collection
            if ids:
                try:
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas
                    )
                    logger.info(f"Added batch of {len(ids)} chunks to vector store")
                except Exception as e:
                    logger.error(f"Error adding batch to vector store: {str(e)}")

    def add_from_embeddings_file(self, embeddings_file: str) -> None:
        """Add the document from an embeddings' file.

        Args:
            embeddings_file: Path to embeddings JSON file
        """
        try:
            document = load_json(embeddings_file)
            self.add_document(document)
        except Exception as e:
            logger.error(f"Error loading embeddings file {embeddings_file}: {str(e)}")

    def add_all_from_directory(self, embeddings_dir: str = "data/embeddings") -> None:
        """Add all documents from embeddings directory.

        Args:
            embeddings_dir: Directory containing embeddings' files
        """
        if not os.path.exists(embeddings_dir):
            logger.error(f"Embeddings directory not found: {embeddings_dir}")
            return

        embedding_files = [f for f in os.listdir(embeddings_dir)
                           if f.endswith(".json") and not f.startswith("embedding_summary")]

        if not embedding_files:
            logger.error(f"No embedding files found in {embeddings_dir}")
            return

        logger.info(f"Adding {len(embedding_files)} documents to vector store")

        for embedding_file in tqdm(embedding_files, desc="Processing embedding files"):
            file_path = os.path.join(embeddings_dir, embedding_file)
            self.add_from_embeddings_file(file_path)

    def query(self, query_embedding: List[float],
              n_results: int = 5,
              filter_criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the vector store for similar documents.

        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            filter_criteria: Optional filter to apply to query

        Returns:
            Dictionary with query results
        """
        logger.info(f"Querying vector store with {len(query_embedding)} dimensional vector")

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_criteria
            )

            return results
        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            return {"ids": [], "distances": [], "documents": [], "metadatas": []}

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            # Get total count
            count = self.collection.count()

            # Get sample entries to analyze metadata structure
            sample = self.collection.peek(limit=5)

            # Get unique document IDs
            if sample["metadatas"]:
                unique_docs = set()
                for metadata in sample["metadatas"]:
                    if "document_id" in metadata:
                        unique_docs.add(metadata["document_id"])

                stats = {
                    "total_chunks": count,
                    "sample_metadata_keys": list(sample["metadatas"][0].keys()) if sample["metadatas"] else [],
                    "sample_unique_documents": len(unique_docs)
                }
            else:
                stats = {"total_chunks": count}

            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}


if __name__ == "__main__":
    # Example usage
    vector_store = VectorStore()

    # Add all documents from embeddings directory
    vector_store.add_all_from_directory()

    # Print collection stats
    stats = vector_store.get_collection_stats()
    logger.info(f"Vector store collection stats: {stats}")
