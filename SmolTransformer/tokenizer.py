from transformers.models.albert import AlbertTokenizer
from config import ModelArgs

class Tokenizer:
    def __init__(self, hf_token=None):
        """Initialize tokenizer with optional HF token"""
        self.hf_token = hf_token
        self.tokenizer = None
        
    def ready_tokenizer(self):
        """Prepare and return the tokenizer"""
        if self.tokenizer is None:
            self.tokenizer = AlbertTokenizer.from_pretrained("ai4bharat/IndicBARTSS")
        return self.tokenizer

def initialize_tokenizer(hf_token=None):
    """Initialize tokenizer with optional HF token"""
    tokenizer_instance = Tokenizer(hf_token=hf_token)
    return tokenizer_instance.ready_tokenizer()

# Global tokenizer instance
tokenizer = None

def get_tokenizer():
    """Get the global tokenizer instance"""
    global tokenizer
    if tokenizer is None:
        tokenizer = initialize_tokenizer()
    return tokenizer
