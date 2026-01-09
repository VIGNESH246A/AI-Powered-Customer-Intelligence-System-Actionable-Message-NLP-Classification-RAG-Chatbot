import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Clean and normalize text data"""
    
    def __init__(self):
        pass
    
    def clean_text(self, text: str) -> str:
        """Apply all cleaning operations"""
        text = self.remove_extra_whitespace(text)
        text = self.normalize_unicode(text)
        text = self.fix_line_breaks(text)
        return text
    
    def remove_extra_whitespace(self, text: str) -> str:
        """Remove excessive whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        return text.strip()
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters"""
        # Replace common unicode quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        # Replace em/en dashes
        text = text.replace('—', '-').replace('–', '-')
        return text
    
    def fix_line_breaks(self, text: str) -> str:
        """Fix inconsistent line breaks"""
        # Ensure section headers have proper spacing
        text = re.sub(r'(Section \d+:)', r'\n\n\1', text)
        # Ensure proper spacing after colons in headers
        text = re.sub(r'(\w+:)(\w)', r'\1 \2', text)
        return text
    
    def remove_special_chars(self, text: str, keep_chars: str = "") -> str:
        """Remove special characters except specified ones"""
        pattern = f"[^a-zA-Z0-9\s{re.escape(keep_chars)}]"
        return re.sub(pattern, '', text)
    
    def preprocess_document(self, text: str) -> str:
        """Main preprocessing pipeline"""
        logger.info("Starting text preprocessing...")
        cleaned = self.clean_text(text)
        logger.info(f"Preprocessing complete. Length: {len(cleaned)}")
        return cleaned


def preprocess_text(text: str) -> str:
    """Convenience function for preprocessing"""
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess_document(text)