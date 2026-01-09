import json
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and manage knowledge base documents"""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        
    def load_text_file(self) -> str:
        """Load content from a text file"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Loaded {len(content)} characters from {self.file_path}")
            return content
        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            raise
    
    def load_json_file(self) -> List[Dict]:
        """Load content from a JSON file"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} items from {self.file_path}")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            raise
    
    def save_json(self, data: List[Dict], output_path: Path) -> None:
        """Save data to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(data)} items to {output_path}")
        except Exception as e:
            logger.error(f"Error saving JSON: {e}")
            raise


def load_knowledge_base(file_path: Path) -> str:
    """Convenience function to load knowledge base"""
    loader = DataLoader(file_path)
    return loader.load_text_file()