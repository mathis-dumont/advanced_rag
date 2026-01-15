# Advanced Multimodal RAG System

## Project Description

An advanced **multimodal RAG (Retrieval Augmented Generation)** system that processes both text and images from technical documentation. Originally designed for maintenance support, this system demonstrates sophisticated RAG techniques including vision-based image understanding, semantic chunking with contextual windows, and incremental knowledge base updates.

**Key Differentiator**: True multimodal processing with AI-generated image descriptions integrated into the semantic search, enabling retrieval based on visual content alongside text.

> **Note**: Word document (`.doc`/`.docx`) input was a technical constraint from the original use case. The system converts these to PDF for unified processing.

The project provides both a CLI for administration and a Streamlit web interface for end-users.

## Table of Contents

1.  [Project Description](#project-description)
2.  [Advanced RAG Features](#advanced-rag-features)
3.  [Key Capabilities](#key-capabilities)
4.  [Installation and Configuration](#installation-and-configuration)
5.  [Usage](#usage)
6.  [Project Structure](#project-structure)
7.  [Technical Implementation](#technical-implementation)

## Advanced RAG Features

This implementation goes beyond basic RAG with several sophisticated techniques:

### 1. **Multimodal Processing**
- **Image extraction** from PDFs using PyMuPDF
- **AI-powered image description** via Mistral's vision API
- **Integrated visual context**: Image descriptions are embedded alongside text, enabling semantic search over visual content
- Parallel processing of multiple images for efficiency

### 2. **Semantic Chunking with Context Windows**
- **Sentence-aware chunking**: Uses spaCy NLP to respect linguistic boundaries
- **Token-precise control**: tiktoken ensures accurate chunk sizing for LLM context limits
- **Overlapping chunks**: Configurable overlap preserves context across boundaries
- **Dynamic context expansion**: Retrieves adjacent chunks (configurable window) to provide broader context to the LLM

### 3. **Incremental Knowledge Base Management**
- **Auto mode**: Automatically detects if rebuild is needed
- **Rebuild mode**: Complete reindexing from scratch
- **Incremental mode**: Processes only new documents
- **Deduplication**: Tracks processed files to avoid redundant work

### 4. **Production-Grade Vector Search**
- **FAISS optimization**: IndexFlatIP with L2 normalization for cosine similarity
- **Efficient retrieval**: Sub-millisecond search even with thousands of chunks
- **Configurable k-NN**: Adjustable number of neighbors for recall/precision tuning

### 5. **Source Traceability**
- Every response includes citations with document name and page number
- Enables verification and fact-checking
- Maintains trust in AI-generated responses

## Key Capabilities

*   **True multimodal RAG**: Processes both text and images from documents
*   **Production-ready pipeline**: Handles document conversion, chunking, embedding, indexing, and retrieval
*   **Flexible deployment**: CLI for administration, Streamlit UI for end users
*   **State-of-the-art models**: Mistral for embeddings, vision, and generation
*   **Configurable architecture**: JSON-based settings for easy experimentation
*   **Incremental updates**: Add new documents without rebuilding entire index

## Project Structure

The project follows a `src-layout` structure for clean separation between source code and other files.

```

├── data/
│   ├── input/                # Place your .doc/.docx documents here
│   ├── static/               # Generated PDF files for web access
│   └── vector_store/         # Vector database (created automatically)
├── src/
│   ├── rag_agent/            # Main RAG application package
│   │   ├── components/       # RAG modules (chunker, retriever...)
│   │   ├── io/               # Input/output modules (loaders, converters)
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── pipeline.py
│   ├── app.py                # CLI entry point
│   └── streamlit_chat.py     # Streamlit UI entry point
├── .env                      # File for secrets (API key)
├── .gitignore
├── README.md
├── requirements.txt
└── settings.json             # Main configuration file
```

## Installation and Configuration

Follow these steps to set up the environment.

### Prerequisites

1.  **Python** (version 3.10+ recommended).
2.  **LibreOffice**: Must be installed on the system.
    *   On Linux (Debian/Ubuntu): `sudo apt-get install libreoffice`
    *   On macOS/Windows: Install from the official website and ensure the `soffice` executable is in your system's PATH.

### Installation Steps

1.  **Clone the repository**
    ```bash
    git clone <repository_url>
    cd advanced_rag_project
    ```

2.  **Cadvanced_rag_projectent and activate it**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the spaCy language model**
    ```bash
    python -m spacy download fr_core_news_md
    ```

5.  **Configure the Mistral API Key**
    Create a `.env` file at the project root and add your key:
    ```env
    MISTRAL_API_KEY=YOUR_MISTRAL_API_KEY_HERE
    ```
    
    **Important:** Do not use quotes around the API key value. The key should be written directly without quotes.

6.  **Create data directories and add your documents**
    ```bash
    mkdir -p data/input
    ```
    
    **Place your `.doc` or `.docx` documents in `data/input/`**
    
    The system will automatically:
    - Convert them to PDF (stored in `static/`)
    - Extract text and images
    - Build the vector database (stored in `data/vector_store/`)

7.  **Configure the Project** (Optional)
    Check and adapt the `settings.json` file at the root if necessary. The default configuration works out of the box.
    ```json
    {
      "data_dir": "data/input",
      "pdf_output_dir": "static",
      "db_dir": "data/vector_store",
      "nlp_model": "fr_core_news_md",
      "tokenizer_encoding": "cl100k_base",
      "embedding_model": "mistral-embed",
      "chat_model": "mistral-medium-2505",
      "chunk_max_tokens": 350,
      "chunk_overlap": 40,
      "k": 10,
      "window": 1,
      "process_images": true
    }
    ```

## Usage

### Launch the Web Interface (Streamlit)

1. **Ensure your documents are in `data/input/`**

2. **Start the Streamlit app:**

```bash
streamlit run src/streamlit_chat.py --server.fileWatcherType none
```

On first launch, the application will automatically:
-   Convert your Word documents to PDF
-   Extract text and images with AI descriptions
-   Build the vector database

This may take a few minutes depending on document volume.

### Command-Line Interface (CLI)

The CLI is ideal for administration, testing, and automation.

*   **Simple interactive mode:**
    ```bash
    python src/app.py
    ```
*   **Force database rebuild:**
    ```bash
    python src/app.py --mode rebuild
    ```
*   **Update the database with new files:**
    ```bash
    python src/app.py --mode incremental
    ```
*   **Ask a question directly:**
    ```bash
    python src/app.py --query "How do I configure the authentication system?"
    ```

## Project Structure

```
├── data/
│   ├── input/                # Place your .doc/.docx documents here
│   ├── static/               # Generated PDF files
│   └── vector_store/         # Vector database (auto-generated)
├── src/
│   ├── rag_agent/
│   │   ├── components/       # Core RAG components
│   │   │   ├── chunking.py       # Semantic text segmentation
│   │   │   ├── embedding.py      # Vector generation
│   │   │   ├── index_manager.py  # FAISS operations
│   │   │   └── retriever.py      # Similarity search + context windows
│   │   ├── io/
│   │   │   ├── converters.py     # Word → PDF conversion
│   │   │   └── loaders.py        # Multimodal PDF processing
│   │   ├── config.py         # Configuration management
│   │   └── pipeline.py       # RAG orchestration
│   ├── app.py                # CLI interface
│   └── streamlit_chat.py     # Web interface
├── .env                      # API keys (not in git)
├── requirements.txt
└── settings.json             # Pipeline configuration
```

## Technical Implementation

### Multimodal Pipeline
The system uses Mistral's vision API to generate descriptions of embedded images, which are then integrated into the text chunks before embedding. This allows the vector search to retrieve relevant content based on visual elements, not just text.

### Contextual Window Retrieval
Instead of returning isolated chunks, the retriever fetches `k` most similar chunks plus a configurable window of adjacent chunks. This provides the LLM with broader context, improving coherence in responses.

### Incremental Indexing
The system tracks processed files and supports incremental updates, making it practical for evolving document bases without expensive full rebuilds.

### Production Considerations
- **Caching**: Heavy objects (spaCy model, Chunker) are cached using Streamlit session state
- **Error handling**: Comprehensive logging and graceful degradation
- **Configurability**: All hyperparameters exposed via `settings.json`
- **Source attribution**: Every response includes document citations

## Future Enhancements

1.  **Hybrid search**: Combine dense (vector) and sparse (BM25) retrieval
2.  **Reranking**: Add cross-encoder for precision improvement
3.  **Query expansion**: LLM-based query reformulation
4.  **Metadata filtering**: Extract and use document structure (sections, headers)
5.  **Evaluation framework**: Automated quality metrics (RAGAS, BLEU, etc.)
6.  **Scalable indexing**: HNSW or IVF for larger corpora (>1M chunks)