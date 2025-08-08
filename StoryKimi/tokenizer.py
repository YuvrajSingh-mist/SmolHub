from transformers import AutoTokenizer

class Tokenizer:
    
    def __init__(self, hf_token=None) -> None:
        # Try to get token from environment if not provided
        
        if hf_token:
            print(f"[INFO] Using HF token for model access")
        else:
            print("[INFO] No HF token provided - using public models only")
            
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=hf_token)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def ready_tokenizer(self):
        
        return self.tokenizer
