"""Text chunking utilities for RAG pipeline.

This module provides semantic text chunking functionality with token-based
segmentation while respecting sentence boundaries.
"""
import logging
from dataclasses import dataclass
from typing import List, Tuple

import tiktoken

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk extracted from a document page.
    
    Attributes:
        text: Text content of the chunk
        file: Source filename
        page: Page number in the PDF
    """
    text: str
    file: str
    page: int


class Chunker:
    """Splits text into semantic chunks with token-based size control.
    
    Uses spaCy for sentence segmentation and tiktoken for accurate token counting.
    Implements lazy loading for the spaCy model to improve startup performance.
    """
    
    def __init__(self,
                 nlp_model: str = "fr_core_news_md",
                 token_model: str = "cl100k_base",
                 max_tokens: int = 350,
                 overlap: int = 40):
        """Initialize the Chunker with specified models and parameters.
        
        Args:
            nlp_model: spaCy model name for sentence segmentation
            token_model: tiktoken model name for tokenization
            max_tokens: Maximum tokens per chunk
            overlap: Number of overlapping tokens between consecutive chunks
        """
        # Store model names only; actual loading is deferred
        self.nlp_model = nlp_model
        self._nlp = None  # spaCy model loaded on first use
        
        try:
            self.enc = tiktoken.encoding_for_model(token_model)
        except Exception:
            self.enc = tiktoken.get_encoding(token_model)
        
        self.max_tokens = max_tokens
        self.overlap = overlap
        logger.info("Chunker initialized with model=%s, max_tokens=%d, overlap=%d",
                   nlp_model, max_tokens, overlap)

    @property
    def nlp(self):
        """Lazy-load spaCy model on first access.
        
        The model is loaded only once and cached for subsequent calls.
        
        Returns:
            Loaded spaCy language model
            
        Raises:
            OSError: If the spaCy model is not installed
        """
        if self._nlp is None:
            logger.debug("Loading spaCy model: %s", self.nlp_model)
            import spacy
            try:
                self._nlp = spacy.load(self.nlp_model)
                logger.info("spaCy model loaded successfully: %s", self.nlp_model)
            except OSError as e:
                logger.error("spaCy model '%s' not found. Install with: python -m spacy download %s",
                           self.nlp_model, self.nlp_model)
                raise OSError(
                    f"spaCy model '{self.nlp_model}' not found. "
                    f"Install with: python -m spacy download {self.nlp_model}"
                ) from e
        return self._nlp

    def split_into_chunks(
        self,
        text: str,
        file: str,
        page: int
    ) -> List[Chunk]:
        """Split page text into token-based chunks while respecting sentence boundaries.
        
        Args:
            text: Text content to split
            file: Source filename for metadata
            page: Page number for metadata
            
        Returns:
            List of Chunk objects containing text segments
        """
        # Trigger lazy loading of spaCy model on first call
        sentences = [sent.text.strip() for sent in self.nlp(text).sents if sent.text.strip()]
        
        chunks: List[Chunk] = []
        current_tokens: List[int] = []

        for sent in sentences:
            token_ids = self.enc.encode(sent)
            
            # Start new chunk if adding this sentence exceeds max_tokens
            if len(current_tokens) + len(token_ids) > self.max_tokens:
                chunk_text = self.enc.decode(current_tokens)
                chunks.append(Chunk(text=chunk_text, file=file, page=page))
                
                # Apply overlap for context preservation
                if self.overlap > 0:
                    current_tokens = current_tokens[-self.overlap:]
                else:
                    current_tokens = []
            
            current_tokens.extend(token_ids)

        # Add remaining tokens as final chunk
        if current_tokens:
            chunk_text = self.enc.decode(current_tokens)
            chunks.append(Chunk(text=chunk_text, file=file, page=page))

        return chunks

    def chunk_all(
        self,
        docs: List[Tuple[str, str, int]]
    ) -> List[Chunk]:
        """Apply chunking across multiple documents.
        
        Args:
            docs: List of tuples (text, file, page)
            
        Returns:
            List of Chunk objects from all documents
        """
        all_chunks: List[Chunk] = []
        for text, file, page in docs:
            all_chunks.extend(self.split_into_chunks(text, file, page))
        
        logger.debug("Chunked %d documents into %d chunks", len(docs), len(all_chunks))
        return all_chunks