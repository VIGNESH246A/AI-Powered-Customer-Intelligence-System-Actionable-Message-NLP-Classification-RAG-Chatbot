import logging
from typing import Dict
from ..actionable_classifier import ActionableClassifier
from ..rag_system.rag_pipeline import RAGPipeline
from ..rag_system.vector_store import FAISSVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedPipeline:
    """
    Unified pipeline combining actionable detection and RAG chatbot
    
    Workflow:
    1. User sends message
    2. Actionable classifier determines if message needs response
    3. If actionable -> RAG chatbot generates response
    4. If non-actionable -> Returns acknowledgment message
    """
    
    def __init__(self, 
                 vector_store: FAISSVectorStore,
                 classifier_model_type: str = 'random_forest',
                 models_dir: str = 'models/actionable_detection'):
        """
        Initialize unified pipeline
        
        Args:
            vector_store: FAISS vector store for RAG
            classifier_model_type: 'random_forest' or 'lstm'
            models_dir: Directory containing classifier models
        """
        # Initialize actionable classifier
        self.classifier = ActionableClassifier(
            model_type=classifier_model_type,
            models_dir=models_dir
        )
        
        # Initialize RAG pipeline
        self.rag_pipeline = RAGPipeline(vector_store)
        
        logger.info("Unified Pipeline initialized successfully")
    
    def process_message(self, message: str) -> Dict:
        """
        Process incoming customer message
        
        Args:
            message: Raw customer message
            
        Returns:
            dict with:
                - 'is_actionable': bool
                - 'classification': dict with confidence scores
                - 'response': str (RAG response if actionable, else acknowledgment)
                - 'rag_context': dict (only if actionable)
        """
        logger.info(f"Processing message: {message[:50]}...")
        
        # Step 1: Classify if message is actionable
        classification = self.classifier.predict(message)
        
        is_actionable = classification['is_actionable']
        confidence = classification['confidence']
        
        logger.info(f"Classification: {'Actionable' if is_actionable else 'Non-actionable'} "
                   f"(Confidence: {confidence:.2%})")
        
        # Step 2: Generate appropriate response
        if is_actionable:
            # Message needs response - use RAG chatbot
            rag_result = self.rag_pipeline.query(message)
            
            return {
                'is_actionable': True,
                'classification': classification,
                'response': rag_result['response'],
                'rag_context': {
                    'context_chunks': rag_result['context_chunks'],
                    'num_chunks_retrieved': rag_result['num_chunks_retrieved']
                }
            }
        else:
            # Non-actionable message - return acknowledgment
            acknowledgment = self._generate_acknowledgment(message)
            
            return {
                'is_actionable': False,
                'classification': classification,
                'response': acknowledgment,
                'rag_context': None
            }
    
    def process_message_with_scores(self, message: str) -> Dict:
        """
        Process message with detailed scores and context
        
        Returns additional relevance scores for retrieved chunks
        """
        logger.info(f"Processing message with scores: {message[:50]}...")
        
        # Classify
        classification = self.classifier.predict(message)
        is_actionable = classification['is_actionable']
        
        if is_actionable:
            # Use RAG with scores
            rag_result = self.rag_pipeline.query_with_scores(message)
            
            return {
                'is_actionable': True,
                'classification': classification,
                'response': rag_result['response'],
                'rag_context': {
                    'context_chunks': rag_result['context_chunks'],
                    'relevance_scores': rag_result['relevance_scores'],
                    'num_chunks_retrieved': rag_result['num_chunks_retrieved']
                }
            }
        else:
            acknowledgment = self._generate_acknowledgment(message)
            
            return {
                'is_actionable': False,
                'classification': classification,
                'response': acknowledgment,
                'rag_context': None
            }
    
    def _generate_acknowledgment(self, message: str) -> str:
        """
        Generate acknowledgment for non-actionable messages
        
        Args:
            message: Original message
            
        Returns:
            Acknowledgment string
        """
        # Simple acknowledgments based on message content
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['thank', 'thanks', 'appreciate']):
            return "You're welcome! I'm glad I could help. Feel free to reach out if you need anything else!"
        
        elif any(word in message_lower for word in ['love', 'great', 'amazing', 'excellent']):
            return "Thank you for your positive feedback! We're thrilled to hear you're satisfied with our service!"
        
        elif any(word in message_lower for word in ['hi', 'hello', 'hey']):
            return "Hello! How can I assist you today? Feel free to ask about our products, policies, or any issues you're experiencing."
        
        else:
            return "Thank you for your message! If you have any questions or need assistance, please don't hesitate to ask."
    
    def clear_conversation(self):
        """Clear RAG conversation history"""
        self.rag_pipeline.clear_history()
        logger.info("Conversation history cleared")
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        rag_stats = self.rag_pipeline.get_stats()
        
        return {
            'classifier_model': self.classifier.model_type,
            'rag_stats': rag_stats,
            'vector_store_stats': rag_stats['vector_store_stats'],
            'conversation_length': rag_stats['conversation_length']
        }


def create_unified_pipeline(vector_store: FAISSVectorStore,
                           classifier_model: str = 'random_forest') -> UnifiedPipeline:
    """
    Convenience function to create unified pipeline
    
    Args:
        vector_store: FAISS vector store
        classifier_model: 'random_forest' or 'lstm'
    
    Returns:
        UnifiedPipeline instance
    """
    return UnifiedPipeline(vector_store, classifier_model_type=classifier_model)