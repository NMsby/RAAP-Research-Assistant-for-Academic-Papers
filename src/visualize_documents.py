import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import logging
from dotenv import load_dotenv

from embedding_service import EmbeddingService
from vector_store import VectorStore
from utils import load_json, ensure_directory

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def visualize_document_relationships():
    """Visualize relationships between documents using embeddings."""
    logger.info("Loading document embeddings...")

    # Create output directory
    output_dir = "visualizations"
    ensure_directory(output_dir)

    # Load embeddings
    embeddings_dir = "data/embeddings"
    embedding_files = [f for f in os.listdir(embeddings_dir)
                       if f.endswith(".json") and not f.startswith("embedding_summary")]

    if not embedding_files:
        logger.error("No embedding files found")
        return

    # Process each document
    all_embeddings = []
    document_titles = []
    document_indices = []

    for file_idx, embedding_file in enumerate(embedding_files):
        file_path = os.path.join(embeddings_dir, embedding_file)
        document = load_json(file_path)

        title = document.get("metadata", {}).get("title", embedding_file)
        document_titles.append(title)

        # Sample embeddings (use first embedding from each document)
        for i, chunk in enumerate(document.get("chunks", [])[:5]):  # Take the first 5 chunks
            if "embedding" in chunk:
                all_embeddings.append(chunk["embedding"])
                document_indices.append(file_idx)

    # Convert to a numpy array
    embeddings_array = np.array(all_embeddings)

    # Apply dimensionality reduction with t-SNE
    logger.info("Applying t-SNE for dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings) - 1))
    embeddings_2d = tsne.fit_transform(embeddings_array)

    # Create visualization
    plt.figure(figsize=(12, 8))

    # Create a color map
    colors = plt.cm.rainbow(np.linspace(0, 1, len(document_titles)))

    # Plot each document's embeddings
    for i, title in enumerate(document_titles):
        # Get indices for this document
        indices = [j for j, doc_idx in enumerate(document_indices) if doc_idx == i]

        # Plot points
        plt.scatter(
            embeddings_2d[indices, 0],
            embeddings_2d[indices, 1],
            color=colors[i],
            label=title[:50] + "..." if len(title) > 50 else title,
            alpha=0.7
        )

    plt.title("Document Embedding Relationships")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(loc='best')
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, "document_relationships.png")
    plt.savefig(output_path)
    logger.info(f"Visualization saved to {output_path}")

    # Show plot if in interactive environment
    plt.close()


if __name__ == "__main__":
    visualize_document_relationships()
