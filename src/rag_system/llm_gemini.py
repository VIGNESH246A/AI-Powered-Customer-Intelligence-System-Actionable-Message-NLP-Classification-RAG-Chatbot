import logging
from google import genai
from google.genai import types
from .config import GOOGLE_API_KEY, GEMINI_MODEL, MAX_TOKENS, TEMPERATURE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
client = genai.Client(api_key=GOOGLE_API_KEY)


class GeminiLLM:
    """Wrapper for Google Gemini LLM"""
    
    def __init__(self, model_name: str = GEMINI_MODEL, 
                 temperature: float = TEMPERATURE,
                 max_tokens: int = MAX_TOKENS):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = client
        
        logger.info(f"Initialized Gemini LLM: {model_name}")
    
    def generate_response(self, prompt: str) -> str:
        """Generate response from prompt"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    top_p=0.95,
                    top_k=40
                )
            )
            
            if hasattr(response, 'text') and response.text:
                return response.text
            elif hasattr(response, 'candidates') and response.candidates:
                return response.candidates[0].content.parts[0].text
            else:
                logger.error(f"Unexpected response format: {response}")
                return "I apologize, but I received an unexpected response format."
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble processing your request. Please try again."
    
    def generate_streaming_response(self, prompt: str):
        """Generate streaming response"""
        try:
            response = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    top_p=0.95,
                    top_k=40
                )
            )
            for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield "I apologize, but I'm having trouble processing your request."
    
    def chat(self, messages: list) -> str:
        """Chat with conversation history"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=messages[-1]['content'],
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            
            if hasattr(response, 'text') and response.text:
                return response.text
            elif hasattr(response, 'candidates') and response.candidates:
                return response.candidates[0].content.parts[0].text
            else:
                return "I apologize, but I received an unexpected response format."
                
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return "I apologize, but I'm having trouble processing your request."


class ConversationManager:
    """Manage conversation history"""
    
    def __init__(self, max_history: int = 10):
        self.history = []
        self.max_history = max_history
    
    def add_user_message(self, message: str):
        """Add user message to history"""
        self.history.append({
            'role': 'user',
            'content': message
        })
        self._trim_history()
    
    def add_assistant_message(self, message: str):
        """Add assistant message to history"""
        self.history.append({
            'role': 'assistant',
            'content': message
        })
        self._trim_history()
    
    def _trim_history(self):
        """Keep only recent messages"""
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_history_string(self) -> str:
        """Format history as string"""
        history_parts = []
        for msg in self.history:
            role = "Customer" if msg['role'] == 'user' else "Assistant"
            history_parts.append(f"{role}: {msg['content']}")
        return "\n\n".join(history_parts)
    
    def clear(self):
        """Clear conversation history"""
        self.history = []


def create_llm() -> GeminiLLM:
    """Convenience function to create LLM instance"""
    return GeminiLLM()