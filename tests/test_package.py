from smolhub.scripts.finetune import SFTTrainer
from smolhub.scripts.lora import LoRAModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from smolhub.helper.dataset.dataset_main import PreprocessDataset
from smolhub.helper.scheduler import CustomLRScheduler
from tests.load_config import Config
import torch

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