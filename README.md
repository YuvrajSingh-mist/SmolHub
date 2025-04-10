# SmolHub

A lightweight package for fine-tuning language models using LoRA (Low-Rank Adaptation).

## Installation

```bash
pip install smolhub
```

## Usage

```python
import torch
import smolhub
# from smolhub.helper.dataset.load_config import Config
from smolhub.scripts.finetune import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from smolhub.helper.scheduler import CustomLRScheduler
from smolhub.scripts.lora import LoRAModel
from smolhub.helper.dataset.dataset_main import PreprocessDataset
from load_config import Config #Needs to be created

model_id = "openai-community/gpt2"

config = Config().get_config()
dataset_path = config["Dataset"]["dataset_path"]

tokenizer = AutoTokenizer.from_pretrained(model_id, token=config['huggingface']['hf_token'])
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", token=config['huggingface']['hf_token'])

if tokenizer.pad_token is None:
    # Set the pad token to the eos token if it doesn't exist
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # tokenizer.pad_token = tokenizer.eos_token
    

    print("Setting pad token as PAD token ")

model.resize_token_embeddings(len(tokenizer))

lora_model = LoRAModel(model)
optimizer = torch.optim.Adam(lora_model.parameters(), lr=2e-3)
scheduler = CustomLRScheduler(optimizer, warmup_iters=100, lr_decay_iters=2000, min_lr=2e-5, max_lr=2e-3, _type="cosine")

#Loading the dataset
preprocess_dataset = PreprocessDataset(dataset_path=dataset_path, tokenizer=tokenizer)
train_dataloader, val_dataloader, test_dataloader = preprocess_dataset.prepare_dataset()

#Initialize the Trainer
sft_trainer = SFTTrainer(lora_model, train_dataloader, val_dataloader, test_dataloader, optimizer, None, scheduler)

#Train
sft_trainer.train()


```

### Config File 

```python 

import yaml


class Config:
    def __init__(self, config_path='/mnt/c/Users/yuvra/OneDrive/Desktop/Work/pytorch/SmolHub/tests/config.yaml'):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def get_config(self):
        return self.config


```