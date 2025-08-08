from transformers import AutoTokenizer
import os

class Tokenizer:
    
    def __init__(self, hf_token=None) -> None:
        # Try to get token from environment if not provided
        if hf_token is None:
            hf_token = os.environ.get('HF_TOKEN')
        
        # Handle default token placeholder
        if hf_token and hf_token != '...':
            print(f"[INFO] Using HF token for model access")
        else:
            print("[INFO] No HF token provided - using public models only")
            hf_token = None
            
        # For Mixtral, we use the Mistral tokenizer instead of Llama
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=hf_token)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def ready_tokenizer(self):
        """Return the tokenizer instance - maintains compatibility with StoryKimi interface"""
        return self.tokenizer

    def get_vocab_size(self):
        """Get vocabulary size"""
        return len(self.tokenizer.get_vocab())

    def encode(self, text, **kwargs):
        """Encode text to tokens"""
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, tokens, **kwargs):
        """Decode tokens to text"""
        return self.tokenizer.decode(tokens, **kwargs)
    
    def __call__(self, *args, **kwargs):
        """Make the class callable to match HuggingFace tokenizer interface"""
        return self.tokenizer(*args, **kwargs)
