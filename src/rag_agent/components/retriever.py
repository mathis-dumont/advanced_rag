"""Document retrieval for RAG pipeline.

This module implements semantic search using FAISS vector indices
with contextual window expansion for improved results.
"""
import logging
from typing import List, Optional, Tuple

import faiss

from .chunking import Chunk
from .embedding import Embedder
from .index_manager import IndexManager

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves relevant document chunks based on semantic similarity."""
    
    def __init__(self,
                 embedder: Embedder,
                 index_manager: IndexManager,
                 k: int = 10,
                 window: int = 1):
        """Initialize the Retriever with necessary components.
        
        Args:
            embedder: Embedder instance for converting queries to vectors
            index_manager: IndexManager instance for loading index and chunks
            k: Number of nearest neighbors to retrieve
            window: Number of adjacent chunks on each side for context expansion
        """
        self.embedder = embedder
        self.index_manager = index_manager
        self.k = k
        self.window = window
        logger.info("Retriever initialized with k=%d, window=%d", k, window)

    def retrieve(self,
                 question: str,
                 k: Optional[int] = None,
                 window: Optional[int] = None
                 ) -> Tuple[List[str], List[str]]:
        """Search for the most relevant passages for the given question.
        
        Args:
            question: Query string
            k: Number of neighbors to retrieve (default: self.k)
            window: Size of adjacent context (default: self.window)
            
        Returns:
            Tuple of (passages, citations) where:
                - passages: List of text passages with context
                - citations: List of source citations (file and page)
        """
        # Use provided parameters or fall back to instance defaults
        k = k if k is not None else self.k
        window = window if window is not None else self.window

        # Load index and chunks
        index, chunks = self.index_manager.load()
        logger.debug("Loaded index with %d chunks for retrieval", len(chunks))

        # Encode the question
        q_vec = self.embedder.embed_texts([question])
        faiss.normalize_L2(q_vec)

        # Search for k nearest neighbors
        distances, indices = index.search(q_vec, k)
        logger.debug("Retrieved %d nearest neighbors", k)

        passages: List[str] = []
        citations: List[str] = []
        
        for idx in indices[0]:
            # Define context window range
            start = max(0, idx - window)
            end = idx + window + 1
            
            # Concatenate texts from context window
            context_chunks = chunks[start:end]
            passages.append("\n".join(c.text for c in context_chunks))
            
            # Citation from pivot chunk
            pivot = chunks[idx]
            citations.append(f"{pivot.file} p.{pivot.page}")

        logger.info("Retrieved %d passages with context window=%d", len(passages), window)
        return passages, citations
