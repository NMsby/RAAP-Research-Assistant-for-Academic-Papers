# src/test_rag_pipeline.py
import os
import argparse
import logging
from dotenv import load_dotenv

from embedding_service import EmbeddingService
from vector_store import VectorStore
from query_engine import QueryEngine

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_embeddings():
    """Process documents and create embeddings."""
    logger.info("Creating embeddings for all processed documents...")
    embedding_service = EmbeddingService()
    result = embedding_service.create_all_document_embeddings()
    logger.info(
        f"Embedding generation complete. {result['successful']}/{result['total']} documents processed successfully.")
    return result['successful'] > 0


def setup_vector_store():
    """Set up vector store with embeddings."""
    logger.info("Setting up vector store...")
    vector_store = VectorStore()
    vector_store.add_all_from_directory()
    stats = vector_store.get_collection_stats()
    logger.info(f"Vector store stats: {stats}")
    return stats.get("total_chunks", 0) > 0


def test_questions(engine):
    """Test the query engine with sample questions."""
    test_queries = [
        "What is Retrieval Augmented Generation?",
        "How does RAG compare to fine-tuning approaches?",
        "What are the main components of a RAG system?",
        "What challenges exist in implementing RAG systems?",
        "How is RAG evaluated for performance?"
    ]

    logger.info("Testing query engine with sample questions...")

    for i, query in enumerate(test_queries):
        logger.info(f"Query {i + 1}: {query}")
        result = engine.answer_with_sources(query)
        print(f"\nQ: {query}")
        print(f"A: {result['formatted_response']}")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Test the RAG pipeline")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation")
    parser.add_argument("--skip-vector-store", action="store_true", help="Skip vector store setup")
    parser.add_argument("--query", type=str, help="Single query to test (optional)")

    args = parser.parse_args()

    # Step 1: Create embeddings (if not skipped)
    if not args.skip_embeddings:
        if not process_embeddings():
            logger.error("Embedding generation failed, exiting.")
            return 1

    # Step 2: Setup vector store (if not skipped)
    if not args.skip_vector_store:
        if not setup_vector_store():
            logger.error("Vector store setup failed, exiting.")
            return 1

    # Step 3: Initialize query engine
    logger.info("Initializing query engine...")
    engine = QueryEngine()

    # Step 4: Test
    if args.query:
        # Run a single query
        result = engine.answer_with_sources(args.query)
        print(f"\nQ: {args.query}")
        print(f"A: {result['formatted_response']}")
    else:
        # Run the test set
        test_questions(engine)

    logger.info("RAG pipeline test complete!")
    return 0


if __name__ == "__main__":
    main()
