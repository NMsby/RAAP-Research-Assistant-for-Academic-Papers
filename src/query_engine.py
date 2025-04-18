import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import time
import google.generativeai as genai
from dotenv import load_dotenv

from embedding_service import EmbeddingService
from vector_store import VectorStore
from utils import ensure_directory, save_json, load_json

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueryEngine:
    """Engine for processing queries and generating responses with RAG."""

    def __init__(self,
                 model_name: str = "gemini-1.5-flash",
                 embedding_model: str = "models/text-embedding-004",
                 api_key: Optional[str] = None,
                 vector_collection: str = "research_papers"):
        """Initialize the query engine.

        Args:
            model_name: Name of the generation model
            embedding_model: Name of the embedding model
            api_key: Google API key (defaults to GOOGLE_API_KEY env variable)
            vector_collection: Name of the vector collection
        """
        # Configure API
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("No API key provided. Set GOOGLE_API_KEY environment variable.")

        genai.configure(api_key=api_key)

        # Initialize models
        self.model = genai.GenerativeModel(model_name)
        self.embedding_service = EmbeddingService(api_key=api_key, model_name=embedding_model)
        self.vector_store = VectorStore(collection_name=vector_collection)

        logger.info(f"Query engine initialized with models: {model_name} and {embedding_model}")

    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a query.

        Args:
            query: Query text

        Returns:
            Query embedding vector
        """
        return self.embedding_service.generate_embedding(query, task_type="retrieval_query")

    def retrieve_context(self, query_embedding: List[float],
                         n_results: int = 5,
                         filter_criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Retrieve relevant context for a query.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filter_criteria: Optional filter criteria

        Returns:
            Dictionary with retrieved context
        """
        return self.vector_store.query(query_embedding, n_results, filter_criteria)

    def format_context(self, context_results: Dict[str, Any]) -> str:
        """Format context results into a string for the prompt.

        Args:
            context_results: Results from the vector store query

        Returns:
            Formatted context string
        """
        if not context_results.get("documents"):
            return "No relevant context found."

        context_str = ""

        # Get all lists from results
        documents = context_results.get("documents", [[]])[0]
        metadatas = context_results.get("metadatas", [[]])[0]
        distances = context_results.get("distances", [[]])[0]

        # Format each retrieved chunk
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            title = meta.get("document_title", "Untitled")
            section = meta.get("section", "")

            # Format with metadata
            context_str += f"\n--- CONTEXT PASSAGE {i + 1} ---\n"
            context_str += f"Source: {title}\n"
            if section:
                context_str += f"Section: {section}\n"
            context_str += f"Relevance: {1.0 - dist:.2f}\n\n"
            context_str += doc.strip()
            context_str += "\n\n"

        return context_str

    def generate_prompt(self, query: str, context: str) -> str:
        """Generate a prompt for the LLM.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Formatted prompt string
        """
        return f"""Please answer the following question based on the provided context from research papers. 
If the answer cannot be determined from the context, say so clearly.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

    def generate_response(self, prompt: str) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: Formatted prompt

        Returns:
            Generated response
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def answer_question(self, query: str, n_results: int = 5,
                        filter_criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Answer a question using RAG.

        Args:
            query: User query
            n_results: Number of context passages to retrieve
            filter_criteria: Optional filter criteria for retrieval

        Returns:
            Dictionary with the query, context, and response
        """
        logger.info(f"Processing query: {query}")

        # Track timing
        start_time = time.time()

        # Step 1: Generate query embedding
        query_embedding = self.generate_query_embedding(query)
        embedding_time = time.time() - start_time

        # Step 2: Retrieve relevant context
        context_results = self.retrieve_context(query_embedding, n_results, filter_criteria)
        retrieval_time = time.time() - start_time - embedding_time

        # Step 3: Format context
        formatted_context = self.format_context(context_results)

        # Step 4: Generate prompt
        prompt = self.generate_prompt(query, formatted_context)

        # Step 5: Generate response
        response = self.generate_response(prompt)
        generation_time = time.time() - start_time - embedding_time - retrieval_time

        # Create the result
        result = {
            "query": query,
            "context": {
                "documents": context_results.get("documents", [[]]),
                "metadatas": context_results.get("metadatas", [[]]),
                "relevance": [1.0 - d for d in context_results.get("distances", [[]])[0]] if context_results.get(
                    "distances") else []
            },
            "response": response,
            "timing": {
                "embedding": embedding_time,
                "retrieval": retrieval_time,
                "generation": generation_time,
                "total": time.time() - start_time
            }
        }

        logger.info(f"Query processed in {result['timing']['total']:.2f} seconds")
        return result

    def answer_with_sources(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Answer a question with cited sources.

        Args:
            query: User query
            n_results: Number of context passages to retrieve

        Returns:
            Dictionary with the query, formatted response with citations, and source details
        """
        # Get the regular answer
        result = self.answer_question(query, n_results)

        # Extract sources for citation
        sources = []
        if result["context"]["metadatas"] and result["context"]["metadatas"][0]:
            for i, metadata in enumerate(result["context"]["metadatas"][0]):
                title = metadata.get("document_title", "Untitled")
                relevance = result["context"]["relevance"][i] if i < len(result["context"]["relevance"]) else 0

                sources.append({
                    "title": title,
                    "section": metadata.get("section", ""),
                    "relevance": relevance,
                    "authors": metadata.get("authors", []),
                    "document_id": metadata.get("document_id", "")
                })

        # Add sources to result
        result["sources"] = sources

        # Generate a formatted response with citations
        if sources:
            formatted_response = result["response"] + "\n\nSources:\n"
            for i, source in enumerate(sources):
                formatted_response += f"[{i + 1}] {source['title']}"
                if source.get("authors"):
                    if isinstance(source["authors"], list):
                        formatted_response += f" - {', '.join(source['authors'])}"
                    else:
                        formatted_response += f" - {source['authors']}"
                formatted_response += "\n"

            result["formatted_response"] = formatted_response
        else:
            result["formatted_response"] = result["response"]

        return result

    def answer_multiple_choice(self, question: str, choices: List[str], n_results: int = 5) -> Dict[str, Any]:
        """Answer a multiple choice question using RAG.

        Args:
            question: Question text
            choices: List of possible answers
            n_results: Number of context passages to retrieve

        Returns:
            Dictionary with answer, confidence, and reasoning
        """
        # Format the multiple choice question
        formatted_question = question + "\n\nOptions:\n"
        for i, choice in enumerate(choices):
            formatted_question += f"{chr(65 + i)}. {choice}\n"

        # Create a specialized prompt for multiple choice
        query_embedding = self.generate_query_embedding(question)  # Embed just the question
        context_results = self.retrieve_context(query_embedding, n_results)
        formatted_context = self.format_context(context_results)

        prompt = f"""Please answer the following multiple choice question based on the provided context from research papers.
First, analyze the question and each option carefully.
Then, explain your reasoning for each option.
Finally, select the best answer choice by letter.

CONTEXT:
{formatted_context}

QUESTION:
{formatted_question}

REASONING:"""

        response = self.generate_response(prompt)

        # Extract the chosen answer (assume it's at the end of the response)
        answer_letter = None
        valid_letters = [chr(65 + i) for i in range(len(choices))]

        for letter in valid_letters:
            if f"Answer: {letter}" in response or f"answer is {letter}" in response or f"choose {letter}" in response.lower():
                answer_letter = letter
                break

        result = {
            "question": question,
            "choices": choices,
            "response": response,
            "chosen_letter": answer_letter,
            "chosen_answer": choices[ord(answer_letter) - 65] if answer_letter else None,
            "context": {
                "documents": context_results.get("documents", [[]]),
                "metadatas": context_results.get("metadatas", [[]])
            }
        }

        return result


if __name__ == "__main__":
    # Example usage
    engine = QueryEngine()

    # Test a simple query
    query = "What are the key components of a RAG system?"
    result = engine.answer_with_sources(query)

    print(f"Query: {query}")
    print(f"Response:\n{result['formatted_response']}")
    print(f"Processing time: {result['timing']['total']:.2f} seconds")
