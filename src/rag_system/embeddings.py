import numpy as np
import logging
from typing import List
from google import genai
from .config import GOOGLE_API_KEY, EMBEDDING_MODEL
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
client = genai.Client(api_key=GOOGLE_API_KEY)


class EmbeddingGenerator:
    """Generate embeddings using Google Gemini Embedding API"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.client = client
        logger.info(f"Initialized embedding generator with model: {model_name}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=[text]
            )
            embedding = np.array(result.embeddings[0].values, dtype=np.float32)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_batch_embeddings(self, texts: List[str], 
                                  batch_size: int = 5,
                                  delay: float = 2.0) -> np.ndarray:
        """Generate embeddings for multiple texts with rate limiting"""
        embeddings = []
        
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        logger.info("This may take a while due to API rate limits...")
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            
            for text in batch:
                retry_count = 0
                max_retries = 3
                
                while retry_count < max_retries:
                    try:
                        embedding = self.generate_embedding(text)
                        embeddings.append(embedding)
                        time.sleep(delay)
                        break
                    except Exception as e:
                        retry_count += 1
                        if "429" in str(e) or "quota" in str(e).lower():
                            wait_time = 30 * retry_count
                            logger.warning(f"Rate limit hit. Waiting {wait_time}s before retry {retry_count}/{max_retries}...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Failed to generate embedding: {e}")
                            embeddings.append(np.zeros(768, dtype=np.float32))
                            break
                
                if retry_count >= max_retries:
                    logger.error(f"Max retries reached for text: {text[:50]}...")
                    embeddings.append(np.zeros(768, dtype=np.float32))
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        return embeddings_array
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a query"""
        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=[query]
            )
            embedding = np.array(result.embeddings[0].values, dtype=np.float32)
            return embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise


def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Convenience function to generate embeddings"""
    generator = EmbeddingGenerator()
    return generator.generate_batch_embeddings(texts)


def generate_query_embedding(query: str) -> np.ndarray:
    """Convenience function to generate query embedding"""
    generator = EmbeddingGenerator()
    return generator.generate_query_embedding(query)