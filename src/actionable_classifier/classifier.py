import joblib
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import logging
import re

from .text_cleaner import TextCleaner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionableClassifier:
    """Enhanced classifier with fallback rules for better detection"""
    
    # Actionable keywords and patterns
    ACTIONABLE_KEYWORDS = {
        'problems': ['not working', 'broken', 'defect', 'damaged', 'stopped', 'failed', 
                    'error', 'issue', 'problem', 'malfunction', 'fault'],
        'requests': ['refund', 'return', 'exchange', 'cancel', 'replace', 'warranty',
                    'help', 'support', 'assist', 'fix', 'repair'],
        'questions': ['how do i', 'how can i', 'how to', 'when will', 'where is',
                     'what is', 'why is', 'can i', 'could you'],
        'complaints': ['disappointed', 'unhappy', 'frustrated', 'angry', 'upset',
                      'terrible', 'awful', 'worst', 'horrible'],
        'urgent': ['urgent', 'asap', 'immediately', 'emergency', 'critical']
    }
    
    # Non-actionable keywords
    NON_ACTIONABLE_KEYWORDS = {
        'praise': ['thank', 'thanks', 'appreciate', 'grateful', 'great', 'excellent',
                  'amazing', 'wonderful', 'fantastic', 'love', 'awesome', 'perfect',
                  'happy', 'satisfied', 'pleased'],
        'general': ['weather', 'day', 'just browsing', 'looking around', 'nice']
    }
    
    def __init__(self, model_type='random_forest', models_dir='models/actionable_detection'):
        """
        Initialize enhanced classifier with fallback rules
        
        Args:
            model_type: 'random_forest' or 'lstm'
            models_dir: Path to model files
        """
        self.model_type = model_type
        self.models_dir = Path(models_dir)
        self.text_cleaner = TextCleaner()
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load trained models"""
        try:
            if self.model_type == 'random_forest':
                tfidf_path = self.models_dir / 'tfidf_vectorizer.pkl'
                self.vectorizer = joblib.load(tfidf_path)
                
                rf_path = self.models_dir / 'random_forest_model.pkl'
                self.model = joblib.load(rf_path)
                
                logger.info("Random Forest model loaded successfully")
                
            elif self.model_type == 'lstm':
                lstm_path = self.models_dir / 'lstm_model.h5'
                self.model = load_model(lstm_path)
                
                self._build_tokenizer()
                
                logger.info("LSTM model loaded successfully")
                
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")
                
        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _build_tokenizer(self):
        """Build tokenizer for LSTM"""
        try:
            tokenizer_path = self.models_dir / 'lstm_tokenizer.pkl'
            if tokenizer_path.exists():
                self.tokenizer = joblib.load(tokenizer_path)
                self.max_len = 20
                logger.info("Loaded pre-saved tokenizer")
                return
        except Exception:
            pass
        
        try:
            df = pd.read_csv('data/nlp_data/messages_cleaned.csv')
            self.tokenizer = Tokenizer(num_words=5000)
            self.tokenizer.fit_on_texts(df['clean_message'])
            self.max_len = 20
            logger.info("Built tokenizer from training data")
        except Exception as e:
            logger.error(f"Could not build tokenizer: {e}")
            raise
    
    def _check_actionable_patterns(self, text: str) -> dict:
        """
        Check for actionable patterns using rule-based approach
        Returns confidence and reasoning
        """
        text_lower = text.lower()
        
        # Count actionable signals
        actionable_score = 0
        non_actionable_score = 0
        
        # Check actionable keywords
        for category, keywords in self.ACTIONABLE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    actionable_score += 2 if category == 'urgent' else 1
        
        # Check non-actionable keywords
        for category, keywords in self.NON_ACTIONABLE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    non_actionable_score += 1
        
        # Check for question marks (strong indicator)
        if '?' in text:
            actionable_score += 2
        
        # Check for negative sentiment words
        negative_words = ['not', 'no', 'never', 'none', 'nobody', 'nothing', "n't", 'without']
        for word in negative_words:
            if word in text_lower:
                actionable_score += 1
        
        # Calculate confidence
        total_score = actionable_score + non_actionable_score
        
        if total_score == 0:
            return {'is_actionable': None, 'confidence': 0.0, 'method': 'rules'}
        
        actionable_confidence = actionable_score / total_score
        
        # Strong signal if confidence is high
        if actionable_confidence >= 0.6:
            return {
                'is_actionable': True,
                'confidence': actionable_confidence,
                'method': 'rules',
                'actionable_score': actionable_score,
                'non_actionable_score': non_actionable_score
            }
        elif actionable_confidence <= 0.4:
            return {
                'is_actionable': False,
                'confidence': 1 - actionable_confidence,
                'method': 'rules',
                'actionable_score': actionable_score,
                'non_actionable_score': non_actionable_score
            }
        
        return {'is_actionable': None, 'confidence': 0.5, 'method': 'rules'}
    
    def predict(self, text: str) -> dict:
        """
        Enhanced prediction with fallback rules
        
        Args:
            text: Raw message text
            
        Returns:
            dict with 'is_actionable', 'confidence', 'label', 'method'
        """
        # First, try rule-based detection
        rule_result = self._check_actionable_patterns(text)
        
        # If rules give strong signal, use that
        if rule_result['is_actionable'] is not None and rule_result['confidence'] > 0.7:
            logger.info(f"Using rule-based classification (confidence: {rule_result['confidence']:.2%})")
            return {
                'is_actionable': rule_result['is_actionable'],
                'confidence': float(rule_result['confidence']),
                'label': int(rule_result['is_actionable']),
                'method': 'rules',
                'probabilities': {
                    'non_actionable': float(1 - rule_result['confidence']) if rule_result['is_actionable'] else float(rule_result['confidence']),
                    'actionable': float(rule_result['confidence']) if rule_result['is_actionable'] else float(1 - rule_result['confidence'])
                }
            }
        
        # Otherwise, use ML model
        cleaned_text = self.text_cleaner.clean_text(text)
        
        if not cleaned_text:
            # Empty after cleaning - use rules
            if rule_result['is_actionable'] is not None:
                return {
                    'is_actionable': rule_result['is_actionable'],
                    'confidence': float(rule_result['confidence']),
                    'label': int(rule_result['is_actionable']),
                    'method': 'rules_fallback',
                    'message': 'Empty after cleaning, used rules'
                }
            return {
                'is_actionable': False,
                'confidence': 0.5,
                'label': 0,
                'method': 'default',
                'message': 'Empty message after cleaning'
            }
        
        # Get ML prediction
        if self.model_type == 'random_forest':
            ml_result = self._predict_rf(cleaned_text)
        elif self.model_type == 'lstm':
            ml_result = self._predict_lstm(cleaned_text)
        
        # If ML model is confident, use it
        if ml_result['confidence'] > 0.8:
            ml_result['method'] = 'ml_model'
            logger.info(f"Using ML model (confidence: {ml_result['confidence']:.2%})")
            return ml_result
        
        # If both have low confidence, prefer rules for actionable detection
        if rule_result['is_actionable'] and rule_result['confidence'] > 0.5:
            logger.info(f"Using rules due to low ML confidence (rule: {rule_result['confidence']:.2%}, ml: {ml_result['confidence']:.2%})")
            return {
                'is_actionable': True,
                'confidence': float(rule_result['confidence']),
                'label': 1,
                'method': 'rules_override',
                'probabilities': {
                    'non_actionable': float(1 - rule_result['confidence']),
                    'actionable': float(rule_result['confidence'])
                }
            }
        
        # Default to ML model
        ml_result['method'] = 'ml_model_default'
        return ml_result
    
    def _predict_rf(self, cleaned_text: str) -> dict:
        """Predict using Random Forest"""
        vector = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(vector)[0]
        probabilities = self.model.predict_proba(vector)[0]
        confidence = probabilities.max()
        
        return {
            'is_actionable': bool(prediction == 1),
            'confidence': float(confidence),
            'label': int(prediction),
            'probabilities': {
                'non_actionable': float(probabilities[0]),
                'actionable': float(probabilities[1])
            }
        }
    
    def _predict_lstm(self, cleaned_text: str) -> dict:
        """Predict using LSTM"""
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(sequence, maxlen=self.max_len)
        
        probability = float(self.model.predict(padded, verbose=0)[0][0])
        prediction = 1 if probability >= 0.5 else 0
        
        return {
            'is_actionable': bool(prediction == 1),
            'confidence': float(probability if prediction == 1 else 1 - probability),
            'label': int(prediction),
            'probabilities': {
                'non_actionable': float(1 - probability),
                'actionable': float(probability)
            }
        }
    
    def batch_predict(self, texts: list) -> list:
        """Predict for multiple texts"""
        return [self.predict(text) for text in texts]