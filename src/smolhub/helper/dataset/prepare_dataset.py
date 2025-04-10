import torch
from torch.utils.data import Dataset
import pandas as pd
from datasets import Dataset
from smolhub.helper.dataset.load_config import Config
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchEncoding


config = Config().get_config()

def json_to_csv(json_dataset):
    # Convert JSON dataset to pd.DataFrame
    csv_dataset = pd.DataFrame(json_dataset)
    return csv_dataset

def _load_dataset(dataset_path):
    # Load dataset from the given path and convert it to a HF Dataset
    if(dataset_path.endswith(".csv")):
        dataset = pd.read_csv(dataset_path)
        hf_dataset = Dataset.from_pandas(dataset)
        return hf_dataset
    elif(dataset_path.endswith(".json")):
        dataset = pd.read_json(dataset_path)
        csv_dataset = json_to_csv(dataset)
        hf_dataset = Dataset.from_pandas(csv_dataset)
        return hf_dataset
    elif(dataset_path.endswith(".xlsx")):
        dataset = pd.read_excel(dataset_path)
        hf_dataset = Dataset.from_pandas(dataset)
        return hf_dataset
    else:
        raise ValueError("Unsupported dataset format")




    
    

def sft_collate_fn(batch_samples, tokenizer, config, labels):
    """
    Collate function for SFTDataset that:
    1. Handles dynamic padding
    2. Applies loss masking (only compute loss on label portion)
    3. Prepares inputs for causal language modeling
    """
    batch = {
        'input_ids': [],
        
        'labels': [],
        'loss_mask': []
    }
    
    for sample in batch_samples:
        # print(sample)
        text = sample['text']
        label = sample['label']
        
            
        # Format the prompt based on task type
        if config['Dataset']['type'] == "classification":
            
            cls_format = """
                Here is a text:
                Text: {}

                You are tasked to do classification.
                
                
                SOLUTION
                The correct answer is: {}
                
                """
            full_prompt = cls_format.format(text, label)
            prompt_before_label = cls_format.format(text, "")
            # print(full_prompt)
            # print(prompt_before_label)
        else:
            raise ValueError("Unsupported dataset type")

        # Tokenize full prompt and the part before label
        tokenized_text = tokenizer(
            full_prompt,
            truncation=True,
            # padding='max_length',
            # max_length=config["Dataset"]["max_length"],
            return_tensors='pt'
        )
        
        tokenized_prompt_before_label = tokenizer(
            prompt_before_label,
            truncation=True,
            # padding='max_length',
            # max_length=config["Dataset"]["max_length"],
            return_tensors='pt'
        )

        # Calculate where the label starts
        label_start_idx = len(tokenized_prompt_before_label['input_ids'])
        
        # Create loss mask (1 for label tokens, 0 otherwise)
        loss_mask = torch.zeros_like(tokenized_text['input_ids'])
        loss_mask[:, label_start_idx:] = 1
        
        # Prepare labels for causal LM (shifted right)
        labels = tokenized_text['input_ids'].clone()
        labels[:, :-1] = tokenized_text['input_ids'][:, 1:]
        labels[:, -1] = tokenizer.eos_token_id

        # Store in batch
        batch['input_ids'].append(tokenized_text['input_ids'].squeeze(0))
        batch['labels'].append(labels.squeeze(0))
        batch['loss_mask'].append(loss_mask.squeeze(0))

    # Pad sequences to longest in batch
    batch['input_ids'] = pad_sequence(
        batch['input_ids'],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    batch['labels'] = pad_sequence(
        batch['labels'],
        batch_first=True,
        padding_value=tokenizer.pad_token_id  
    )
    batch['loss_mask'] = pad_sequence(
        batch['loss_mask'],
        batch_first=True,
        padding_value=0
    )
    # batch['input_ids'] = torch.stack(batch['input_ids'])
    # batch['labels'] = torch.stack(batch['labels'])
    # batch['loss_mask'] = torch.stack(batch['loss_mask'])
    # # Convert to BatchEncoding
    return BatchEncoding(batch)


def convert_dataset_to_dataloader(dataset, tokenizer, labels):
    # Convert HF Dataset to PyTorch DataLoader

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["Dataset"]["batch_size"], num_workers=config["Dataset"]["num_workers"], shuffle=config["Dataset"]["shuffle"], drop_last=config["Dataset"]["drop_last"], pin_memory=config["Dataset"]["pin_memory"], persistent_workers=config["Dataset"]["persistent_workers"], collate_fn = lambda batch: sft_collate_fn(batch, tokenizer, config, labels))
  
    return dataloader

class SFTDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = _load_dataset(dataset_path)
    

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]












