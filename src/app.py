# src/app.py

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from rag_agent.components.chunking import Chunker
from rag_agent.config import load_settings
from rag_agent.pipeline import RAGPipeline

# Logging configuration for the CLI application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main entry point for the command-line interface."""
    
    # Load environment variables (e.g., MISTRAL_API_KEY) from a .env file
    load_dotenv()
    
    # 1. Load configuration from settings.json
    try:
        settings = load_settings(Path("settings.json"))
    except FileNotFoundError as e:
        logging.error("CRITICAL ERROR: %s. Please create this file.", e)
        return

    # 2. Handle command-line arguments
    parser = argparse.ArgumentParser(description="Document-based RAG system.")
    parser.add_argument(
        "--mode",
        choices=["auto", "rebuild", "incremental"],
        default="auto",
        help="Database update mode (auto, rebuild, incremental)."
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Convert Word documents to PDF without building the index."
    )
    parser.add_argument("--query", type=str, help="Question to ask directly.")
    parser.add_argument(
        "--process-images",
        action=argparse.BooleanOptionalAction,
        default=settings.process_images,
        help="Enable/Disable AI-based image processing."
    )
    args = parser.parse_args()

    # Update settings with CLI arguments
    settings.process_images = args.process_images
    
    # 3. Initialize the pipeline (centralized and clean logic)
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        logging.error("CRITICAL ERROR: The MISTRAL_API_KEY environment variable is not defined.")
        return
        
    try:
        pipeline = RAGPipeline(settings=settings, api_key=api_key)
    except Exception as e:
        logging.error("Fatal error during pipeline initialization: %s", e, exc_info=True)
        return

    # Initialize the Chunker
    chunker = Chunker(
        nlp_model=settings.nlp_model,
        token_model=settings.tokenizer_encoding,
        max_tokens=settings.chunk_max_tokens,
        overlap=settings.chunk_overlap
    )

    # 4. Execute application logic
    if args.convert_only:
        logging.info("'convert-only' mode enabled.")
        # Conversion logic is now handled by the loader, but we can force it here
        for doc in pipeline.loader.data_dir.rglob("*.[dD][oO][cC]*"):
            try:
                pdf = pipeline.converter.to_pdf(doc)
                logging.info(f"✅ Converted: {doc.name} → {pdf.name}")
            except Exception as e:
                logging.error(f"⚠️ Conversion failed {doc.name}: {e}")
        return

    if args.query:
        logging.info("'query' mode: answering a single question.")
        response = pipeline.answer(args.query, chunker=chunker, update_mode=args.mode)
        print("\n--- Response ---\n")
        print(response)
        return

    # Default interactive mode
    if args.mode != "auto":
        pipeline.build_or_update(chunker=chunker, mode=args.mode)

    print("\n--- Interactive Technical Assistant ---")
    print("Type 'q' or 'exit' to quit.")
    while True:
        try:
            q = input("\nQuestion › ")
            if q.lower() in ("q", "exit", "quit"):
                break
            response = pipeline.answer(q, chunker=chunker)
            print("\n--- Response ---\n")
            print(response)
        except (KeyboardInterrupt, EOFError):
            break
    print("\nSession ended.")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()