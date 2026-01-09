import re
import logging
from typing import List, Dict
from .config import CHUNK_SIZE, CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentChunker:
    """Split documents into overlapping chunks"""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_section(self, text: str) -> List[str]:
        """Split document by sections first"""
        sections = []
        
        # Split by section headers
        pattern = r'(Section \d+:.*?)(?=Section \d+:|$)'
        matches = re.finditer(pattern, text, re.DOTALL)
        
        for match in matches:
            section_text = match.group(1).strip()
            if section_text:
                sections.append(section_text)
        
        logger.info(f"Found {len(sections)} sections")
        return sections
    
    def chunk_by_size(self, text: str) -> List[str]:
        """Split text into fixed-size chunks with overlap"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Define chunk end
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < text_length:
                # Look for period, question mark, or exclamation within last 100 chars
                search_start = max(start, end - 100)
                sentence_end = max(
                    text.rfind('.', search_start, end),
                    text.rfind('!', search_start, end),
                    text.rfind('?', search_start, end)
                )
                
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = end - self.overlap if end < text_length else text_length
        
        return chunks
    
    def create_chunks_with_metadata(self, text: str) -> List[Dict[str, any]]:
        """Create chunks with metadata"""
        chunks = []
        
        # First try section-based chunking
        sections = self.chunk_by_section(text)
        
        if sections:
            # Chunk each section
            for idx, section in enumerate(sections):
                # Extract section title
                title_match = re.match(r'(Section \d+:.*?)(?:\n|$)', section)
                section_title = title_match.group(1) if title_match else f"Section {idx+1}"
                
                # Chunk the section if it's too long
                if len(section) > self.chunk_size:
                    section_chunks = self.chunk_by_size(section)
                    for chunk_idx, chunk in enumerate(section_chunks):
                        chunks.append({
                            "chunk_id": len(chunks),
                            "text": chunk,
                            "section": section_title,
                            "chunk_index": chunk_idx,
                            "char_count": len(chunk)
                        })
                else:
                    chunks.append({
                        "chunk_id": len(chunks),
                        "text": section,
                        "section": section_title,
                        "chunk_index": 0,
                        "char_count": len(section)
                    })
        else:
            # Fallback to simple size-based chunking
            simple_chunks = self.chunk_by_size(text)
            for idx, chunk in enumerate(simple_chunks):
                chunks.append({
                    "chunk_id": idx,
                    "text": chunk,
                    "section": "General",
                    "chunk_index": idx,
                    "char_count": len(chunk)
                })
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks


def chunk_document(text: str, chunk_size: int = CHUNK_SIZE, 
                   overlap: int = CHUNK_OVERLAP) -> List[Dict[str, any]]:
    """Convenience function to chunk document"""
    chunker = DocumentChunker(chunk_size, overlap)
    return chunker.create_chunks_with_metadata(text)