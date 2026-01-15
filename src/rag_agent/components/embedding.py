"""Text embedding generation using Mistral API.

This module provides functionality to convert text into vector embeddings
for semantic similarity search.
"""
import logging
from typing import List

import numpy as np
from mistralai import Mistral

logger = logging.getLogger(__name__)


class Embedder:
    """Generates embeddings for text using Mistral's embedding models."""
    
    def __init__(self,
                 api_key: str,
                 model: str = "mistral-embed",
                 batch_size: int = 64):
        """Initialize the Embedder with API credentials and parameters.
        
        Args:
            api_key: Mistral API key for authentication
            model: Name of the embedding model to use
            batch_size: Batch size for API calls (optimizes throughput)
        """
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        logger.info("Embedder initialized with model=%s, batch_size=%d", model, batch_size)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            NumPy array of shape (len(texts), embedding_dim) containing embeddings
            
        Raises:
            Exception: If API call fails
        """
        vectors: List[List[float]] = []
        
        # Process in batches to optimize API calls
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    inputs=batch
                ).data
                
                # Extract embeddings from response
                for d in response:
                    vectors.append(d.embedding)
                    
                logger.debug("Embedded batch %d/%d (%d texts)", 
                           i // self.batch_size + 1,
                           (len(texts) + self.batch_size - 1) // self.batch_size,
                           len(batch))
            except Exception as e:
                logger.error("Failed to embed batch starting at index %d: %s", i, e)
                raise
        
        # Convert to NumPy array
        result = np.array(vectors, dtype="float32")
        logger.info("Generated embeddings for %d texts (dimension=%d)", 
                   len(texts), result.shape[1] if result.size > 0 else 0)
        return result
