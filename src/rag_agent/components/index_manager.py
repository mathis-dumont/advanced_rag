"""FAISS index management for vector storage and retrieval.

This module handles creation, saving, and loading of FAISS vector indices
along with their associated text chunks.
"""
import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from .chunking import Chunk

logger = logging.getLogger(__name__)


class IndexManager:
    """Manages FAISS index and associated chunks for vector similarity search."""
    
    def __init__(self,
                 db_dir: Path,
                 dim: int = 1024):
        """Initialize the IndexManager with storage configuration.
        
        Args:
            db_dir: Directory for storing index and chunks
            dim: Embedding dimension (for creating empty index)
        """
        self.db_dir = db_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.db_dir / "faiss_index.idx"
        self.chunks_path = self.db_dir / "chunks.pkl"
        self.dim = dim
        logger.info("IndexManager initialized with db_dir=%s, dim=%d", db_dir, dim)

    def build(self, vectors: np.ndarray) -> faiss.Index:
        """Build a FAISS index from the given vectors.
        
        Args:
            vectors: NumPy array of shape (n_samples, dim)
            
        Returns:
            FAISS index with inner product similarity (after L2 normalization)
        """
        # L2 normalization for cosine similarity
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        
        logger.info("Built FAISS index with %d vectors (dim=%d)", 
                   index.ntotal, vectors.shape[1])
        return index

    def save(self,
             index: faiss.Index,
             chunks: List[Chunk]) -> None:
        """Save FAISS index and chunk list to disk.
        
        Args:
            index: FAISS index to save
            chunks: List of Chunk objects corresponding to the vectors
        """
        # Write FAISS index
        faiss.write_index(index, str(self.index_path))
        
        # Serialize chunks
        with open(self.chunks_path, "wb") as f:
            pickle.dump(chunks, f)
        
        logger.info("Saved database: %d chunks to %s", len(chunks), self.db_dir)

    def load(self) -> Tuple[faiss.Index, List[Chunk]]:
        """Load FAISS index and chunk list from disk.
        
        Returns:
            Tuple of (FAISS index, list of Chunk objects)
            
        Raises:
            FileNotFoundError: If index or chunks file is missing
        """
        if not self.index_path.exists() or not self.chunks_path.exists():
            raise FileNotFoundError(
                f"Index or chunks file missing in {self.db_dir}"
            )
        
        # Read FAISS index
        index = faiss.read_index(str(self.index_path))
        
        # Deserialize chunks
        with open(self.chunks_path, "rb") as f:
            chunks: List[Chunk] = pickle.load(f)
        
        logger.info("Loaded database: %d chunks from %s", len(chunks), self.db_dir)
        return index, chunks
