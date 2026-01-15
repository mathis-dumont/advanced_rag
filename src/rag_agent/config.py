"""Configuration management for RAG pipeline.

This module handles loading and validation of application settings from JSON.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)


class Settings(BaseModel):
    """Application settings for the RAG pipeline.
    
    Attributes:
        data_dir: Directory containing input documents
        pdf_output_dir: Directory for converted PDF files
        db_dir: Directory for vector database storage
        nlp_model: spaCy model name for text processing
        tokenizer_encoding: Tokenizer encoding name (e.g., 'cl100k_base')
        embedding_model: Model name for generating embeddings
        chat_model: Model name for chat/generation
        chunk_max_tokens: Maximum tokens per text chunk
        chunk_overlap: Token overlap between consecutive chunks
        k: Number of nearest neighbors to retrieve
        window: Context window size for adjacent chunks
        process_images: Whether to process images with AI description
    """
    
    data_dir: Path
    pdf_output_dir: Path
    db_dir: Path
    nlp_model: str
    tokenizer_encoding: str
    embedding_model: str
    chat_model: str
    chunk_max_tokens: int
    chunk_overlap: int
    k: int
    window: int
    process_images: bool = True

    @model_validator(mode='after')
    def create_directories(self) -> 'Settings':
        """Create necessary directories if they don't exist.
        
        Returns:
            The Settings instance with directories created
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_output_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directories exist: data=%s, pdf=%s, db=%s", 
                    self.data_dir, self.pdf_output_dir, self.db_dir)
        return self


def load_settings(path: Path = Path("settings.json")) -> Settings:
    """Load and validate settings from a JSON configuration file.
    
    Args:
        path: Path to the JSON configuration file
        
    Returns:
        Validated Settings object
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        ValidationError: If the configuration is invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file '{path}' not found.")
    
    logger.info("Loading settings from: %s", path)
    with path.open('r', encoding='utf-8') as f:
        data: Dict[str, Any] = json.load(f)
    
    settings = Settings.model_validate(data)
    logger.info("Settings loaded successfully")
    return settings