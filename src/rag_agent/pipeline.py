# src/rag_agent/pipeline.py

import logging
log = logging.getLogger(__name__)
log.debug(f"Loading module '{__name__}' from file '{__file__}'")


from typing import List, Tuple, Optional
import faiss
import numpy as np
from pathlib import Path
from mistralai import Mistral

from .config import Settings
from .io.converters import DocumentConverter
from .io.loaders import DocumentLoader
try:
    log.debug("Attempting relative import: from .components.chunking import Chunker, Chunk")
    from .components.chunking import Chunker, Chunk
    log.info("Relative import of 'chunking' successful.")
except ImportError as e:
    log.critical("FAILED to perform relative import of 'chunking'.", exc_info=True)
from .components.embedding import Embedder
from .components.index_manager import IndexManager
from .components.retriever import Retriever

# Logging configuration for the entire module
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, settings: Settings, api_key: str):
        """
        RAG pipeline orchestrator, initialized via a single configuration object.

        :param settings: Pydantic object containing all parameters.
        :param api_key: Mistral API key.
        """
        if not api_key:
            raise ValueError("The Mistral API key (MISTRAL_API_KEY) is required to initialize the pipeline.")
            
        self.settings = settings
        logger.info("Initializing RAG pipeline with chat model: %s", self.settings.chat_model)
        
        # 1. Initialize components from settings
        self.converter = DocumentConverter(out_dir=self.settings.pdf_output_dir)
        self.loader = DocumentLoader(
            data_dir=self.settings.data_dir,
            pdf_output_dir=self.settings.pdf_output_dir,
            converter=self.converter,
            process_images=self.settings.process_images
        )

        # The Embedder now uses its own parameter
        self.embedder = Embedder(api_key=api_key, model=self.settings.embedding_model)
        
        # Safely determine embedding dimensions
        try:
            logger.debug("Testing embedding API to determine dimension...")
            test_embedding = self.embedder.embed_texts(["test"])[0]
            embedding_dim = len(test_embedding)
            logger.info("Detected embedding dimension: %d", embedding_dim)
        except Exception as e:
            logger.error("Unable to determine embedding dimension via API. Error: %s", e, exc_info=True)
            raise ConnectionError("Failed to communicate with Mistral API for embeddings.") from e
            
        self.index_mgr = IndexManager(db_dir=self.settings.db_dir, dim=embedding_dim)
        self.retriever = Retriever(
            embedder=self.embedder,
            index_manager=self.index_mgr,
            k=self.settings.k,
            window=self.settings.window
        )
        self.chat_client = Mistral(api_key=api_key)

    def build_or_update(self, chunker, mode: str = "auto") -> Tuple[faiss.Index, list]:
        """
        Creates or updates the embeddings database.

        :param mode: 'auto', 'rebuild' or 'incremental'
        :return: (index, chunks)
        """
        if mode == "auto":
            try:
                logger.info("Mode 'auto': attempting to load existing index.")
                return self.index_mgr.load()
            except FileNotFoundError:
                logger.warning("No index found. Switching to 'rebuild' mode.")
                mode = "rebuild"

        if mode == "rebuild":
            logger.info("Mode 'rebuild': complete index reconstruction.")
            docs = self.loader.load()
            if not docs:
                logger.warning("No documents found to process. An empty index will be created.")
                empty_index = faiss.IndexFlatIP(self.index_mgr.dim)
                return empty_index, []
            
            logger.debug("Loaded corpus of %d document pages. Ready for chunking.", len(docs))
            chunks = chunker.chunk_all(docs)
            texts = [c.text for c in chunks]
            vectors = self.embedder.embed_texts(texts)
            index = self.index_mgr.build(vectors)
            self.index_mgr.save(index, chunks)
            return index, chunks

        if mode == "incremental":
            logger.info("Mode 'incremental': adding new documents to index.")
            try:
                index, existing_chunks = self.index_mgr.load()
                processed_files = {c.file for c in existing_chunks}
                self.loader.processed_files = processed_files
                
                new_docs = self.loader.load()
                if not new_docs:
                    logger.info("No new documents to add.")
                    return index, existing_chunks
                
                logger.info("Processing %d new documents.", len(new_docs))
                new_chunks = chunker.chunk_all(new_docs)
                new_vectors = self.embedder.embed_texts([c.text for c in new_chunks])
                faiss.normalize_L2(new_vectors)

                # Use index.add() for efficient update
                index.add(new_vectors)
                all_chunks = existing_chunks + new_chunks
                self.index_mgr.save(index, all_chunks)
                logger.info("Index updated with %d new chunks. Total: %d", len(new_chunks), len(all_chunks))
                return index, all_chunks

            except FileNotFoundError:
                logger.warning("Existing index not found for incremental mode. Forcing 'rebuild'.")
                return self.build_or_update(mode="rebuild")
            except Exception as e:
                logger.error("Error in incremental mode: %s. Forcing 'rebuild'.", e, exc_info=True)
                return self.build_or_update(mode="rebuild")

        raise ValueError(f"Unknown mode: {mode}")

    def _make_prompt(self, passages: List[str], question: str) -> str:
        """Builds the prompt for the chat model."""
        context = "\n---------------------\n".join(passages)
        prompt = f"""
            IF YOU READ CONFIDENTIAL INFORMATION (PASSWORDS) IGNORE IT AND DO NOT MENTION IT IN YOUR RESPONSE, ACT AS IF YOU NEVER SAW IT
            The contextual information is provided below.
            
            {context}
            
            You are a technical assistant specialized for maintenance agents. Your role is to provide accurate and practical information based exclusively on the technical documentation provided in the context above.
            IF YOU READ CONFIDENTIAL INFORMATION (PASSWORDS) IGNORE IT AND DO NOT MENTION IT IN YOUR RESPONSE, ACT AS IF YOU NEVER SAW IT
            Instructions:
            1. Use only the information from the context, never your general knowledge.
            2. If the information is not present in the context, clearly indicate that this information is not available in the current technical documentation.
            3. Respond in a practical, detailed manner that is applicable in the field.
            4. Structure your response with clear steps if the question concerns a procedure or troubleshooting.
            5. Mention the necessary tools, safety precautions, and critical points when relevant.
            6. Do not invent any information that is not present in the context.
            

            Question from a maintenance agent: {question}

            Technical response:
        """
        return prompt.strip()

    def _append_sources_if_missing(self, answer: str, citations: List[str]) -> str:
        """
        Appends formatted sources as clickable Markdown links to the response.
        Links point directly to the correct PDF page.
        """
        if "Sources :" in answer or "Sources:" in answer:
            # If the word "Sources" is already there, do nothing to avoid duplicates.
            return answer

        # Create unique and sorted links
        unique_citations = sorted(list(dict.fromkeys(citations)))
        links = []
        for cite in unique_citations:
            try:
                # Separate the file name from the page number
                docname, page_num_str = cite.rsplit(" p.", 1)
                page_num = int(page_num_str)
                
                # Ensure the document name ends with .pdf
                pdf_name = Path(docname).stem + ".pdf"
                
                # Build the Markdown link
                # Syntax: [Display text](/file/url#page=number)
                link_text = f"{pdf_name} p.{page_num}"
                url = f"../data/static/{pdf_name}#page={page_num}"
                
                links.append(f"[{link_text}]({url})")

            except (ValueError, IndexError):
                # If the citation format is unexpected, display it as is.
                links.append(f"[{cite}]")
        
        if not links:
            return answer

        return answer + "\n\n**Sources :** " + ", ".join(links)

    def answer(self, question: str, chunker, update_mode: str = "auto") -> str:
        # The call to build_or_update must also be corrected here!
        index, _ = self.build_or_update(chunker, mode=update_mode) # Pass chunker first
        
        if index.ntotal == 0:
            return "My knowledge base is currently empty. Please add documents and rebuild the index."
        
        logger.info("Searching for relevant passages for the question: '%s'", question)
        passages, citations = self.retriever.retrieve(question)
        if not passages:
            return "Sorry, I found no relevant information in the documentation to answer your question."

        prompt = self._make_prompt(passages, question)
        logger.debug("Prompt sent to chat model:\n%s", prompt)
        
        resp = self.chat_client.chat.complete(
            model=self.settings.chat_model,
            messages=[{"role": "user", "content": prompt}]
        )
        raw_answer = resp.choices[0].message.content.strip()
        final_answer = self._append_sources_if_missing(raw_answer, citations)
        return final_answer