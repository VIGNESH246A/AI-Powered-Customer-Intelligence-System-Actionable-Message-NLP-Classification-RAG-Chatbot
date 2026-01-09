import re
import spacy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextCleaner:
    """Clean and preprocess text for actionable classification"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Spacy model loaded successfully")
        except OSError:
            logger.warning("Spacy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """
        Clean text following the same preprocessing as training
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text ready for classification
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)
        
        # Remove punctuation and numbers
        text = re.sub(r"[^a-z\s]", "", text)
        
        if self.nlp is None:
            # Fallback: simple tokenization without lemmatization
            words = text.split()
            words = [w for w in words if len(w) > 2]
            return " ".join(words)
        
        # Process text with SpaCy
        doc = self.nlp(text)
        
        # Lemmatize and remove stopwords
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and len(token) > 2
        ]
        
        return " ".join(tokens)
    
    def batch_clean(self, texts: list) -> list:
        """Clean multiple texts"""
        return [self.clean_text(text) for text in texts]