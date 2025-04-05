
from datasets import load_dataset
from prepare_dataset import convert_dataset_to_dataloader
from prepare_dataset import SFTDataset
from load_config import Config

config = Config().get_config()

class PreprocessDataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.use_hf_dataset = config["Dataset"]["use_hf_dataset"]
        if(self.use_hf_dataset):
            self.dataset = load_dataset(self.dataset_path)
        else:
            self.sft_dataset = SFTDataset(dataset_path)
    
    def prepare_dataset(self):

        if(self.use_hf_dataset):
            return load_dataset(self.dataset_path)
        else:
            dataloader = convert_dataset_to_dataloader(self.sft_dataset)
            return dataloader