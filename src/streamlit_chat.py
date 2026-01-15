# src/streamlit_chat.py

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (.env) at startup
load_dotenv()

# Import necessary components
from rag_agent.config import load_settings, Settings
from rag_agent.pipeline import RAGPipeline
from rag_agent.components.chunking import Chunker

# Streamlit page configuration
st.set_page_config(page_title="Technical Support Assistant", layout="wide", initial_sidebar_state="auto")
st.title("💬 Technical Support Assistant")
st.caption("Ask questions about the technical documentation. I'll find the answer for you.")

# --- Caching "LIGHTWEIGHT" and "SAFE" objects ---

@st.cache_resource
def get_main_config() -> Settings:
    """Load the main configuration once."""
    return load_settings(Path("settings.json"))

@st.cache_resource
def get_pipeline(_settings: Settings) -> RAGPipeline:
    """
    Create and cache the RAGPipeline (without the Chunker).
    This is an object that Streamlit can cache without issues.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        st.error("Mistral API key not configured.")
        return None
    return RAGPipeline(settings=_settings, api_key=api_key)


# --- MAIN ORCHESTRATION ---
try:
    # 1. Get cached components (safe objects)
    main_settings = get_main_config()
    pipeline = get_pipeline(main_settings)

    # 2. Manage the "HEAVY" object (the Chunker) with st.session_state
    # This is the definitive solution that bypasses Streamlit's cache bug.
    if 'chunker' not in st.session_state:
        with st.spinner("Initializing text analysis component (one time only)..."):
            st.session_state.chunker = Chunker(
                nlp_model=main_settings.nlp_model,
                token_model=main_settings.tokenizer_encoding,
                max_tokens=main_settings.chunk_max_tokens,
                overlap=main_settings.chunk_overlap
            )

    # 3. Warm up the index (if necessary)
    if "index_initialized" not in st.session_state:
        with st.spinner("Preparing the knowledge base..."):
            # Pass the chunker object from session_state
            pipeline.build_or_update(chunker=st.session_state.chunker, mode="auto")
        st.session_state.index_initialized = True
        st.success("Assistant ready!")
        st.rerun()

    # 4. Chat interface logic
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask your question here..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Searching the documentation..."):
                # Pass the chunker object from session_state at each call
                response = pipeline.answer(user_input, chunker=st.session_state.chunker, update_mode="auto")
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

except Exception as e:
    st.error(f"A critical error occurred: {e}")
    st.exception(e) 