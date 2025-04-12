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


def dpo_collate_fn_merged_prompt(batch, tokenizer, device):

    merged_chosen_prompts = []
    merged_rejected_prompts = []

    for sample in batch:

        # print(sample)

        # Extract and merge chosen response
        prompt = sample['prompt']
        chosen_data = sample['chosen']
        chosen_data = "Instruction: " + prompt + "\n" + "Output: " + chosen_data[1]['content'] + "\n"
        # Extract and merge rejected response
        rejected_data = sample['rejected']
        rejected_data =  "Instruction: " + prompt + "\n" + "Output: " + rejected_data[1]['content'] + "\n"

        # print(chosen_data)
        # print(rejected_data)
        merged_chosen_prompts.append(chosen_data)


        merged_rejected_prompts.append(rejected_data)

    tokenized_win_prompt = tokenizer(merged_chosen_prompts, max_length = 1024, padding='max_length', truncation=True, return_tensors="pt").to(device)

    tokenized_lose_prompt = tokenizer(merged_rejected_prompts, max_length = 1024, truncation=True, padding='max_length', return_tensors="pt").to(device)



    return {
        # 'prompt': prompts, # Still return original prompts for potential use
        'chosen': tokenized_win_prompt, # List of merged prompt-chosen texts
        'rejected': tokenized_lose_prompt # List of merged prompt-rejected texts
    }

def convert_dataset_to_dataloader_preference(dataset, tokenizer, device):
    # Convert HF Dataset to PyTorch DataLoader

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["Dataset"]["batch_size"], num_workers=config["Dataset"]["num_workers"], shuffle=config["Dataset"]["shuffle"], drop_last=config["Dataset"]["drop_last"], pin_memory=config["Dataset"]["pin_memory"], persistent_workers=config["Dataset"]["persistent_workers"], collate_fn = lambda batch: dpo_collate_fn_merged_prompt(batch, tokenizer, device))
  
    return dataloader

class PreferenceDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = _load_dataset(dataset_path)
    

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]












