from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from config import ModelArgs
from tokenizer import Tokenizer

def initialize_tokenizer(hf_token=None):
    """Initialize tokenizer with optional HF token"""
    tokenizer_instance = Tokenizer(hf_token=hf_token)
    return tokenizer_instance.ready_tokenizer()

# Default tokenizer (will be updated when called with token)
tokenizer = None

tinystories = True
fw = False
fw_train = None
fw_test = None
if(tinystories):
    
    fw_train = load_dataset("roneneldan/TinyStories", split="train")
    fw_test = load_dataset("roneneldan/TinyStories", split="validation")
    print(fw_train)
    print(fw_test)
if(fw):   
    fw_train = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=False)
    fw_train = fw_train.train_test_split(test_size=0.01)
    print(fw_train)
    print(fw_train)


def prepare_dataset(split, device, batch_size, model_args, tokenizer_instance, use_ddp=False):
    print("Device is: ", device)
    
    global tokenizer
    tokenizer = tokenizer_instance
 
    def collate_fn(batch):
        # Extract text data
        texts = []
        
        for item in batch:
            tt = item['text']# Append EOS token to each text
            texts.append(tt)

        input_encodings = tokenizer(texts, max_length = model_args.block_size,padding='max_length', truncation=True, return_tensors="pt")
        
        input_encodings["labels"] = input_encodings["input_ids"].clone()  # Use `input_ids` as labels

        input_encodings["labels"][:, :-1] = input_encodings["input_ids"][:, 1:]  
        # input_encodings['labels'][:, 0] = tokenizer.bos_token_id
        # input_encodings["labels"][:, -1] = tokenizer.eos_token_id  # Let the last token be end 
       
        return input_encodings

  
    dataloader = None
    if(tinystories):
        if(split == 'train'):
            sampler = DistributedSampler(fw_train, shuffle=True) if use_ddp else None
            shuffle = False if use_ddp else True
            data_loader = DataLoader(
            fw_train,
            # generator=generator,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=shuffle
        )
        elif(split == 'val'):
            sampler = DistributedSampler(fw_test, shuffle=False) if use_ddp else None
            shuffle = False if use_ddp else False
            data_loader = DataLoader(
            fw_test,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=shuffle
        )
    elif(fw):
        if(split == 'train'):
            sampler = DistributedSampler(fw_train['train'], shuffle=True) if use_ddp else None
            shuffle = False if use_ddp else True
            data_loader = DataLoader(
            fw_train['train'],
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=shuffle
    )
        elif(split == 'val'):
            sampler = DistributedSampler(fw_train['test'], shuffle=False) if use_ddp else None
            shuffle = False if use_ddp else False
            data_loader = DataLoader(
            fw_train['test'],
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=shuffle
        )
    return data_loader
