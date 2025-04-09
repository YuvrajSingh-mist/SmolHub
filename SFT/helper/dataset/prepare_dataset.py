import torch
from torch.utils.data import Dataset
import pandas as pd
from datasets import Dataset
from SFT.helper.dataset.load_config import Config

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


def convert_dataset_to_dataloader(dataset, isTrain=False):
    # Convert HF Dataset to PyTorch DataLoader
    if(isTrain):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["Dataset"]["batch_size"], num_workers=config["Dataset"]["num_workers"], shuffle=config["Dataset"]["shuffle"], drop_last=config["Dataset"]["drop_last"], pin_memory=config["Dataset"]["pin_memory"], persistent_workers=config["Dataset"]["persistent_workers"])
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["Dataset"]["batch_size"], num_workers=config["Dataset"]["num_workers"], shuffle=False, drop_last=config["Dataset"]["drop_last"], pin_memory=config["Dataset"]["pin_memory"], persistent_workers=config["Dataset"]["persistent_workers"])
    return dataloader

class SFTDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = _load_dataset(dataset_path)
    

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]












