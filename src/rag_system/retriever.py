import logging
from typing import List, Dict, Tuple
from .embeddings import EmbeddingGenerator
from .vector_store import FAISSVectorStore
from .config import TOP_K_RESULTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextRetriever:
    """Retrieve relevant context for user queries"""
    
    def __init__(self, vector_store: FAISSVectorStore, top_k: int = TOP_K_RESULTS):
        self.vector_store = vector_store
        self.embedding_generator = EmbeddingGenerator()
        self.top_k = top_k
    
    def retrieve(self, query: str) -> List[Dict]:
        """Retrieve top-k relevant chunks for a query"""
        logger.info(f"Retrieving context for query: {query[:50]}...")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(query)
        
        # Search in vector store
        results = self.vector_store.search(query_embedding, k=self.top_k)
        
        # Extract chunks
        chunks = [chunk for chunk, score in results]
        
        logger.info(f"Retrieved {len(chunks)} chunks")
        return chunks
    
    def retrieve_with_scores(self, query: str) -> List[Tuple[Dict, float]]:
        """Retrieve chunks with similarity scores"""
        logger.info(f"Retrieving context with scores for query: {query[:50]}...")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(query)
        
        # Search in vector store
        results = self.vector_store.search(query_embedding, k=self.top_k)
        
        logger.info(f"Retrieved {len(results)} chunks with scores")
        return results
    
    def format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into context string"""
        context_parts = []
        
        for idx, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Context {idx}]")
            context_parts.append(f"Section: {chunk.get('section', 'N/A')}")
            context_parts.append(f"{chunk['text']}")
            context_parts.append("")  # Empty line between chunks
        
        return "\n".join(context_parts)
    
    def retrieve_formatted_context(self, query: str) -> str:
        """Retrieve and format context in one step"""
        chunks = self.retrieve(query)
        return self.format_context(chunks)


def create_retriever(vector_store: FAISSVectorStore) -> ContextRetriever:
    """Convenience function to create retriever"""
    return ContextRetriever(vector_store)