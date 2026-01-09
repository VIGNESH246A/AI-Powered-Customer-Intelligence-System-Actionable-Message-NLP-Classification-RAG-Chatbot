import logging
from typing import Dict, List
from .vector_store import FAISSVectorStore
from .retriever import ContextRetriever
from .llm_gemini import GeminiLLM, ConversationManager
from .prompt_templates import create_rag_prompt, create_followup_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline for customer support"""
    
    def __init__(self, vector_store: FAISSVectorStore):
        self.vector_store = vector_store
        self.retriever = ContextRetriever(vector_store)
        self.llm = GeminiLLM()
        self.conversation_manager = ConversationManager()
        
        logger.info("RAG Pipeline initialized")
    
    def query(self, user_query: str) -> Dict:
        """
        Process user query and generate response
        
        Returns:
            dict with 'response', 'context_chunks', 'num_chunks_retrieved'
        """
        logger.info(f"Processing query: {user_query[:50]}...")
        
        # Retrieve relevant context
        context_chunks = self.retriever.retrieve(user_query)
        context_text = self.retriever.format_context(context_chunks)
        
        # Check if there's conversation history
        if len(self.conversation_manager.history) > 0:
            history_text = self.conversation_manager.get_history_string()
            prompt = create_followup_prompt(user_query, context_text, history_text)
        else:
            prompt = create_rag_prompt(user_query, context_text)
        
        # Generate response
        response = self.llm.generate_response(prompt)
        
        # Update conversation history
        self.conversation_manager.add_user_message(user_query)
        self.conversation_manager.add_assistant_message(response)
        
        return {
            'response': response,
            'context_chunks': context_chunks,
            'num_chunks_retrieved': len(context_chunks)
        }
    
    def query_with_scores(self, user_query: str) -> Dict:
        """
        Process query and return response with relevance scores
        
        Returns:
            dict with 'response', 'context_chunks', 'relevance_scores', 'num_chunks_retrieved'
        """
        logger.info(f"Processing query with scores: {user_query[:50]}...")
        
        # Retrieve context with scores
        results = self.retriever.retrieve_with_scores(user_query)
        context_chunks = [chunk for chunk, score in results]
        scores = [score for chunk, score in results]
        
        context_text = self.retriever.format_context(context_chunks)
        
        # Check if there's conversation history
        if len(self.conversation_manager.history) > 0:
            history_text = self.conversation_manager.get_history_string()
            prompt = create_followup_prompt(user_query, context_text, history_text)
        else:
            prompt = create_rag_prompt(user_query, context_text)
        
        # Generate response
        response = self.llm.generate_response(prompt)
        
        # Update conversation history
        self.conversation_manager.add_user_message(user_query)
        self.conversation_manager.add_assistant_message(response)
        
        return {
            'response': response,
            'context_chunks': context_chunks,
            'relevance_scores': scores,
            'num_chunks_retrieved': len(context_chunks)
        }
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_manager.clear()
        logger.info("Conversation history cleared")
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        return {
            'vector_store_stats': self.vector_store.get_stats(),
            'conversation_length': len(self.conversation_manager.history)
        }


def create_pipeline(vector_store: FAISSVectorStore) -> RAGPipeline:
    """Convenience function to create RAG pipeline"""
    return RAGPipeline(vector_store)