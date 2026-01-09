"""
RAG system module for customer support
"""

from .config import *
from .rag_pipeline import create_pipeline

__all__ = ['create_pipeline']