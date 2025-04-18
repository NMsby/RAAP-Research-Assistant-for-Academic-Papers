# RAAP - Research Assistant for Academic Papers

RAAP is an AI-powered research assistant
that helps users navigate and understand academic papers using Retrieval Augmented Generation
(RAG).
The system can answer questions about papers, generate summaries,
identify research gaps, and find connections between different papers.

## Features

- PDF document processing and metadata extraction
- Generate semantic embeddings for document content
- Store and retrieve document chunks using vector search
- Answer specific questions about paper content
- Paper summarization and comparison
- Research gap identification
- Citation management
- Visualize relationships between research documents

## Components

- **Document Processor**: Extracts text from PDFs and breaks it into meaningful chunks
- **Embedding Service**: Converts text chunks into vector representations
- **Vector Database**: Stores and indexes embeddings for efficient retrieval
- **Query Engine**: Processes questions, finds relevant content, and generates responses with citations

## Technologies

- Google Generative AI (Gemini API)
- ChromaDB for vector storage
- pdfplumber for PDF processing
- scikit-learn for visualization
- Python 3.9+

## Installation

1. Clone this repository
``` 
git clone https://github.com/NMsby/RAAP-Research-Assistant-for-Academic-Papers.git
```
2. Navigate to the project directory
```
cd RAAP-Research-Assistant-for-Academic-Papers
```
3. Install dependencies: `pip install -r requirements.txt`
4. Set up environment variables: Create a `.env` file with the following variables:
```markdown
GOOGLE_API_KEY=your_api_key_here
```

### 1. Document Processing

Process academic papers from PDF to structured chunks:

```bash
# Process a single document
python src/process_single_document.py --file data/raw/your_paper.pdf

# Process a single document with verbose logging
python src/process_single_document_verbose.py --file data/raw/your_paper.pdf

# Process all documents in the data/raw directory
python src/process_all_documents_simple.py
```

### 2. Embedding Generation

Generate embeddings for all processed documents:

```bash
# Generate embeddings for all documents
python src/embedding_service.py
```

### 3. Vector Database Management

```bash
# Reset the vector database (if needed)
python src/reset_vector_db.py

# Build and query the vector database
python src/test_rag_pipeline.py --skip-embeddings
```

### 4. Running Queries

```bash
# Test specific questions related to your papers
python src/test_specific_questions.py

# Test the full RAG pipeline (processing, embeddings, and querying)
python src/test_rag_pipeline.py
```

### 5. Visualization

```bash
# Visualize document relationships
python src/visualize_documents.py

# Visualize the structure of processed documents
python src/visualize_document.py --all
```

## Kaggle Notebook

The project includes a Kaggle notebook that demonstrates the complete system:

1. Access the notebook at https://www.kaggle.com/code/nelsonmasbayi/raap-research-assistant-for-academic-papers
2. Run all cells to see the system in action
3. Use the interactive question-answering widget to ask questions about the papers
4. Explore the document relationship visualization

The notebook provides a user-friendly interface to interact with the research assistant without requiring local setup.

## Adding Your Own Papers

### Local Repository
1. Add PDF files to the `data/raw` directory
2. Create metadata JSON files with the same name (optional)
3. Run the processing and embedding generation scripts
4. Use the query engine to ask questions about your papers

### Kaggle Notebook
1. Upload your papers using the file upload functionality
2. Run the document processing cells
3. Continue through the notebook to generate embeddings and set up the query engine
4. Ask questions about your papers using the interactive widget

## Project Structure

- `src/`: Source code for all components
  - `document_processor.py`: Extracts text from PDFs and creates chunks
  - `embedding_service.py`: Generates embeddings for document chunks
  - `vector_store.py`: Stores and retrieves embeddings
  - `query_engine.py`: Processes queries and generates responses
  - `visualize_documents.py`: Creates visualizations of document relationships
- `data/`: Directory for storing papers and processed data
  - `raw/`: Original PDF files and metadata
  - `processed/`: Processed document chunks
  - `embeddings/`: Document embeddings
  - `vector_db/`: Vector database files
- `notebooks/`: Jupyter notebooks including the Kaggle submission

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
