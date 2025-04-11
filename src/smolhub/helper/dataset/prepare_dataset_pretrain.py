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



def collate_fn(batch_samples, tokenizer, config, labels):
    batch = {
        'input_ids': [],
        'labels': [],
       
    }
    
    for sample in batch_samples:
        # print(sample)
        text = sample['text']

        # Tokenize full prompt and the part before label
        tokenized_text = tokenizer(
            text,
            truncation=True,
            # padding='max_length',
            # max_length=config["Dataset"]["max_length"],
            return_tensors='pt'
        )
        
        # Prepare labels for causal LM (shifted right)
        labels = tokenized_text['input_ids'].clone()
        labels[:, :-1] = tokenized_text['input_ids'][:, 1:]
        labels[:, -1] = tokenizer.eos_token_id

        # Store in batch
        batch['input_ids'].append(tokenized_text['input_ids'].squeeze(0))
        batch['labels'].append(labels.squeeze(0))


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

    return BatchEncoding(batch)


def convert_dataset_to_dataloader_pretrain(dataset, tokenizer, labels):
    # Convert HF Dataset to PyTorch DataLoader

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["Dataset"]["batch_size"], num_workers=config["Dataset"]["num_workers"], shuffle=config["Dataset"]["shuffle"], drop_last=config["Dataset"]["drop_last"], pin_memory=config["Dataset"]["pin_memory"], persistent_workers=config["Dataset"]["persistent_workers"], collate_fn = lambda batch: collate_fn(batch, tokenizer, config, labels))
  
    return dataloader

class PretrainDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = _load_dataset(dataset_path)
    

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]












