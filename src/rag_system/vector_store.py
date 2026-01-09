import faiss
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Dict
from .config import FAISS_INDEX_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS-based vector store for efficient similarity search"""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = None
        self.chunks = []
        self.index_path = Path(FAISS_INDEX_PATH)
        self.index_path.mkdir(parents=True, exist_ok=True)
    
    def create_index(self, embeddings: np.ndarray, chunks: List[Dict]) -> None:
        """Create FAISS index from embeddings"""
        logger.info(f"Creating FAISS index with {len(embeddings)} vectors...")
        
        # Ensure embeddings are float32 and contiguous
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        
        # Normalize embeddings for cosine similarity
        embeddings_norm = embeddings.copy()
        faiss.normalize_L2(embeddings_norm)
        
        # Create index (using IndexFlatIP for cosine similarity)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings_norm)
        self.chunks = chunks
        
        logger.info(f"FAISS index created with {self.index.ntotal} vectors")
    
    def save_index(self) -> None:
        """Save FAISS index and chunks to disk"""
        if self.index is None:
            logger.error("No index to save")
            return
        
        try:
            # Save FAISS index
            index_file = self.index_path / "faiss.index"
            faiss.write_index(self.index, str(index_file))
            
            # Save chunks metadata
            chunks_file = self.index_path / "chunks.pkl"
            with open(chunks_file, 'wb') as f:
                pickle.dump(self.chunks, f)
            
            logger.info(f"Index saved to {self.index_path}")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise
    
    def load_index(self) -> bool:
        """Load FAISS index and chunks from disk"""
        index_file = self.index_path / "faiss.index"
        chunks_file = self.index_path / "chunks.pkl"
        
        if not index_file.exists() or not chunks_file.exists():
            logger.warning("Index files not found")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_file))
            
            # Load chunks metadata
            with open(chunks_file, 'rb') as f:
                self.chunks = pickle.load(f)
            
            logger.info(f"Index loaded with {self.index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Tuple[Dict, float]]:
        """Search for top-k similar chunks"""
        if self.index is None:
            logger.error("Index not initialized")
            return []
        
        # Ensure query embedding is float32 and contiguous
        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query embedding
        query_norm = query_embedding.copy()
        faiss.normalize_L2(query_norm)
        
        # Search
        distances, indices = self.index.search(query_norm, k)
        
        # Return chunks with similarity scores
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks) and idx >= 0:
                results.append((self.chunks[idx], float(distance)))
        
        return results
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "total_chunks": len(self.chunks)
        }


def create_vector_store(embeddings: np.ndarray, chunks: List[Dict]) -> FAISSVectorStore:
    """Convenience function to create and save vector store"""
    store = FAISSVectorStore()
    store.create_index(embeddings, chunks)
    store.save_index()
    return store


def load_vector_store() -> FAISSVectorStore:
    """Convenience function to load existing vector store"""
    store = FAISSVectorStore()
    if store.load_index():
        return store
    else:
        raise FileNotFoundError("Vector store not found. Please create one first.")