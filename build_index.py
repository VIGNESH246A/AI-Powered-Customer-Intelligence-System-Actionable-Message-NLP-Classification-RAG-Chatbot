#!/usr/bin/env python3
"""
Script to build the vector index from knowledge base
Run this once to create the FAISS index before using the chatbot
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.rag_system.config import KB_FILE, CHUNKS_FILE
from src.rag_system.data_loader import load_knowledge_base, DataLoader
from src.rag_system.text_preprocessing import preprocess_text
from src.rag_system.chunking import chunk_document
from src.rag_system.embeddings import generate_embeddings
from src.rag_system.vector_store import create_vector_store
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Build vector index pipeline"""
    
    try:
        # Step 1: Load knowledge base
        logger.info("=" * 60)
        logger.info("Step 1: Loading Knowledge Base")
        logger.info("=" * 60)
        
        if not KB_FILE.exists():
            logger.error(f"Knowledge base file not found: {KB_FILE}")
            logger.info("Please place your knowledge base file at: data/rag_data/raw/kb.txt")
            return
        
        raw_text = load_knowledge_base(KB_FILE)
        logger.info(f"Loaded {len(raw_text)} characters from knowledge base")
        
        # Step 2: Preprocess text
        logger.info("\n" + "=" * 60)
        logger.info("Step 2: Preprocessing Text")
        logger.info("=" * 60)
        
        cleaned_text = preprocess_text(raw_text)
        logger.info(f"Cleaned text length: {len(cleaned_text)} characters")
        
        # Step 3: Chunk document
        logger.info("\n" + "=" * 60)
        logger.info("Step 3: Chunking Document")
        logger.info("=" * 60)
        
        chunks = chunk_document(cleaned_text)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Save chunks to JSON
        loader = DataLoader(CHUNKS_FILE)
        loader.save_json(chunks, CHUNKS_FILE)
        logger.info(f"Saved chunks to {CHUNKS_FILE}")
        
        # Step 4: Generate embeddings
        logger.info("\n" + "=" * 60)
        logger.info("Step 4: Generating Embeddings")
        logger.info("=" * 60)
        logger.info("This may take a few minutes...")
        
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = generate_embeddings(chunk_texts)
        logger.info(f"Generated {len(embeddings)} embeddings")
        logger.info(f"Embedding shape: {embeddings.shape}")
        
        # Step 5: Create and save vector store
        logger.info("\n" + "=" * 60)
        logger.info("Step 5: Creating Vector Store")
        logger.info("=" * 60)
        
        vector_store = create_vector_store(embeddings, chunks)
        stats = vector_store.get_stats()
        logger.info(f"Vector store created successfully!")
        logger.info(f"Stats: {stats}")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("INDEX BUILD COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Total chunks: {len(chunks)}")
        logger.info(f"Total vectors: {stats['total_vectors']}")
        logger.info(f"Vector dimension: {stats['dimension']}")
        logger.info("\nYou can now run the chatbot using: streamlit run app.py")
        
    except Exception as e:
        logger.error(f"Error building index: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()