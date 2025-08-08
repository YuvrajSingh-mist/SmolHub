from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
from config import ModelArgs
from tokenizer import initialize_tokenizer

def load_datasets(token, sample_size=None):
    """Load the datasets with optional sampling for faster analysis"""
    print("Loading datasets...")
    train_dataset = load_dataset("ai4bharat/samanantar", 'hi', split=f"train", token=token)

    print(f"Train dataset loaded")
    
    return train_dataset

def prepare_dataset(split, device, batch_size, model_args, tokenizer_instance, fw_train):
    """Prepare dataset with collate function"""
    print("Device is: ", device)
    
    def collate_fn(batch):
        # Extract text data
        en_texts = []
        hi_texts = []
        
        for item in batch:
            it = tokenizer_instance.bos_token + item['src'] + tokenizer_instance.eos_token 
            en_texts.append(it)
            it = item['tgt'] + tokenizer_instance.eos_token
            hi_texts.append(it)

        input_encodings = tokenizer_instance(en_texts, padding='max_length', max_length=model_args.block_size, truncation=True, add_special_tokens=False, return_tensors="pt")
        target_encodings = tokenizer_instance(hi_texts, padding='max_length', max_length=model_args.block_size, truncation=True, return_tensors="pt", add_special_tokens=False)
        
        input_encodings["labels"] = target_encodings["input_ids"].clone()
        input_encodings["decoder_input_ids"] = target_encodings["input_ids"].clone()
        
        # Shift decoder input ids for teacher forcing
        input_encodings["decoder_input_ids"][:, 1:] = input_encodings["decoder_input_ids"][:, :-1]
        input_encodings["decoder_input_ids"][:, 0] = tokenizer_instance.bos_token_id
        
        return input_encodings

    if split == 'train':
        data_loader = DataLoader(
            fw_train['train'],
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=True,
        )
    elif split == 'val':
        data_loader = DataLoader(
            fw_train['test'],
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=False,
        )
    
    return data_loader
