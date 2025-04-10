
from datasets import load_dataset

from smolhub.helper.dataset.prepare_dataset import SFTDataset, convert_dataset_to_dataloader
from smolhub.helper.dataset.load_config import Config

config = Config().get_config()

class PreprocessDataset:
    def __init__(self, dataset_path=None, tokenizer=None):
        self.dataset_path = dataset_path
        self.use_hf_dataset = config["Dataset"]["use_hf_dataset"]
        # self.tokenizer_path = config["Model"]["tokenizer"]
        self.train = None
        self.val = None
        self.test = None
        self.tokenizer = tokenizer
       
        
    def check_for_split(self, dataset):
        # Initialize all splits to None
        self.train = self.val = self.test = None
        
        # Case 1: Dataset has all three splits
        if all(split in dataset for split in ['train', 'validation', 'test']):
            print("Dataset splits found: train, validation, test")
            self.train = dataset['train']
            self.val = dataset['validation']
            self.test = dataset['test']
            print(self.train)
            print(self.val)
            print(self.test)
            return
        
        # Case 2: Only train exists - create both val and test (10% each)
        if 'train' in dataset and 'validation' not in dataset and 'test' not in dataset:
            print("Creating validation and test splits from train")
            # First split train into temp_train (90%) and test (10%)
            temp_split = dataset['train'].train_test_split(test_size=0.1)
            # Then split temp_train into final train (80%) and val (10%)
            final_split = temp_split['train'].train_test_split(test_size=1/9)  # 0.1/0.9 ≈ 0.111
            self.train = final_split['train']
            self.val = final_split['test']
            self.test = temp_split['test']
            print(self.train)
            print(self.val)
            print(self.test)
            return
        
        # Case 3: Has train and test but no validation - split train to create validation
        if 'train' in dataset and 'test' in dataset and 'validation' not in dataset:
            print("Creating validation split from train")
            self.test = dataset['test']
            # Split train into new train (90%) and validation (10%)
            split = dataset['train'].train_test_split(test_size=0.1)
            self.train = split['train']
            self.val = split['test']
            print(self.train)
            print(self.val)
            print(self.test)
            return
        
        # Case 4: Has train and validation but no test - split train to create test
        if 'train' in dataset and 'validation' in dataset and 'test' not in dataset:
            print("Creating test split from train")
            self.val = dataset['validation']
            # Split train into new train (90%) and test (10%)
            split = dataset['train'].train_test_split(test_size=0.1)
            self.train = split['train']
            self.test = split['test']
            print(self.train)
            print(self.val)
            print(self.test)
            return
        
        # Case 5: Only test exists (unlikely but possible) - use test as validation and split to create new test
        if 'test' in dataset and 'train' not in dataset:
            print("Warning: Only test split found. Using half as validation and splitting to create new test")
            # Split test into val and new test (50% each)
            split = dataset['test'].train_test_split(test_size=0.5)
            self.val = split['train']
            self.test = split['test']
            print(self.train)
            print(self.val)
            print(self.test)
            # No train data available in this case
            return
        
        # Case 6: Only validation exists (unlikely but possible) - use as test and split to create new validation
        if 'validation' in dataset and 'train' not in dataset:
            print("Warning: Only validation split found. Using as test and splitting to create new validation")
            # Split validation into val and test (50% each)
            split = dataset['validation'].train_test_split(test_size=0.5)
            self.val = split['train']
            self.test = split['test']
            print(self.train)
            print(self.val)
            print(self.test)
            # No train data available in this case
            return
        
        # If we get here, the dataset has an unexpected combination of splits
        raise ValueError("Unexpected dataset split configuration. Could not create required splits.")
        
        # A full HF dataset object is required rest it will take care
    def prepare_dataset(self):
        
        if(self.use_hf_dataset):
            self.dataset = load_dataset(self.dataset_path, token=config["huggingface"]["hf_token"])
            # print(self.dataset)
            self.check_for_required_columns(self.dataset)
            self.labels = self.get_unique_labels(self.dataset)
            self.check_for_split(self.dataset)
            self.check_for_required_columns(self.train)
            self.check_for_required_columns(self.test)
            self.check_for_required_columns(self.val)
            
            
            # self.train = self.train.map(self.tokenize)
            # self.val = self.val.map(self.tokenize)
            # self.test = self.test.map(self.tokenize)
            
            # self.train = self.train.remove_columns(["text"])
            # self.val = self.val.remove_columns(["text"])
            # self.test = self.test.remove_columns(["text"])
            
            train_dataloader = convert_dataset_to_dataloader(self.train, self.tokenizer, self.labels)
            val_dataloader = convert_dataset_to_dataloader(self.val, self.tokenizer , self.labels)
            test_dataloader = convert_dataset_to_dataloader(self.test, self.tokenizer   , self.labels)
            
            return train_dataloader, val_dataloader, test_dataloader   
        # If not using HF dataset, we are using a custom dataset
        # Just a full csv or json is required with train key value pair
        else:
            self.sft_dataset = SFTDataset(self.dataset_path)
            self.check_for_required_columns(self.sft_dataset)
            self.labels = self.get_unique_labels(self.sft_dataset)
            self.check_for_split(self.sft_dataset)
            self.check_for_required_columns(self.train)
            self.check_for_required_columns(self.test)
            self.check_for_required_columns(self.val)
            
            # self.train = self.train.map(self.tokenize)
            # self.val = self.val.map(self.tokenize)
            # self.test = self.test.map(self.tokenize)
            
            train_dataloader = convert_dataset_to_dataloader(self.train, self.tokenizer, self.labels)
            val_dataloader = convert_dataset_to_dataloader(self.val, self.tokenizer, self.labels)
            test_dataloader = convert_dataset_to_dataloader(self.test, self.tokenizer, self.labels)
            
            return train_dataloader, val_dataloader, test_dataloader

    
    
    def check_for_required_columns(self, dataset):
        required_columns = ["text", "label"]
        for column in required_columns:
            if column not in dataset.column_names:
                raise ValueError(f"Required column {column} not found in dataset")
   
    def check_label_column(self, dataset):
        if dataset['label'].dtype != 'float32' or dataset['label'].dtype != 'int32':
            raise ValueError("Label column must be of type float or int")


    def get_unique_labels(self, dataset):
        labels = []
        for example in dataset['train']:
            # print(example)
            if example['label'] not in labels:
                labels.append(example['label'])
        return labels
        
