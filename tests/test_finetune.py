import torch
from SFT.helper.dataset.load_config import Config
from SFT.scripts.finetune import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from SFT.helper.scheduler import CustomLRScheduler
from SFT.scripts.lora import LoRAModel
from SFT.helper.dataset.dataset_main import PreprocessDataset

model_id = "openai-community/gpt2"

config = Config().get_config()
dataset_path = config["Dataset"]["dataset_path"]

tokenizer = AutoTokenizer.from_pretrained(model_id, token=config['huggingface']['hf_token'])
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", token=config['huggingface']['hf_token'])

if tokenizer.pad_token is None:
    # Set the pad token to the eos token if it doesn't exist
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print("Setting pad token as '[PAD]")

optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
loss_fn = torch.nn.CrossEntropyLoss()
scheduler = CustomLRScheduler(optimizer, warmup_iters=100, lr_decay_iters=2000, min_lr=2e-5, max_lr=2e-3, _type="cosine")
lora_model = LoRAModel(model)

#Loading the dataset
preprocess_dataset = PreprocessDataset(dataset_path=dataset_path, tokenizer=tokenizer)
train_dataloader = preprocess_dataset.prepare_dataset(isTrain=True)

#Loading the validation dataset
preprocess_dataset = PreprocessDataset(dataset_path=dataset_path, tokenizer=tokenizer)
val_dataloader = preprocess_dataset.prepare_dataset(isTrain=False)

#Initialzie the Trainer
sft_trainer = SFTTrainer( lora_model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler)

#Train
sft_trainer.train()

