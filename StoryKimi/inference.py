import torch
import torch.nn.functional as F
from config import ModelArgs
from model import DeepSeekV3
from tokenizer import Tokenizer

def topk_sampling(model, prompt, device, max_length=50, top_k=50, temperature=1.0, tokenizer=None, hf_token=None):
    if tokenizer is None:
        # Use default tokenizer if none provided
        tokenizer_instance = Tokenizer(hf_token=hf_token)
        tokenizer = tokenizer_instance.ready_tokenizer()
        
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated_tokens = []
    
    if(len(input_ids[0]) < max_length):
        max_length -= len(input_ids[0]) # If the input is longer than max_length, set max_length to the length of the input
    else:
        max_length = len(input_ids[0]) - max_length  
    for _ in range(max_length):
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Pass inference=True to use the inference path in the model
            outputs = model(input_ids, inference=True)
            logits = outputs[:, -1, :]
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            
            # Top-k filtering
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            
            # Sample from top-k
            next_token = torch.multinomial(top_k_probs, num_samples=1)
            
            xcol = torch.gather(top_k_indices, -1, next_token)
            input_ids = torch.cat([input_ids, xcol], dim=1) #1 because is it the dimension of the sequence
            
            if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id and xcol.item() == tokenizer.eos_token_id:
                break
            
            
    return tokenizer.decode(input_ids[0])


def save_text(file_path, step, text):
    with open(file_path, 'w') as f:
        f.write(f"Step {step}: {text}\n")
