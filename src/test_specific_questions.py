import os
import logging
from dotenv import load_dotenv
from query_engine import QueryEngine

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_specific_questions():
    """Test the query engine with questions about the processed papers."""
    engine = QueryEngine()

    test_queries = [
        "What are the main findings about inflation discussed in the papers?",
        "Explain the concept of gravitational phase space discussed in the quantum area fluctuations paper.",
        "What are bigeodisics in dynamical last passage percolation?",
        "Summarize the findings on 3D hand motion and contacts synthesis",
        "What mathematical concepts are discussed in the papers about set families?"
    ]

    logger.info("Testing query engine with paper-specific questions...")

    for i, query in enumerate(test_queries):
        logger.info(f"Query {i + 1}: {query}")
        result = engine.answer_with_sources(query)
        print(f"\nQ: {query}")
        print(f"A: {result['formatted_response']}")
        print("-" * 80)


if __name__ == "__main__":
    test_specific_questions()
