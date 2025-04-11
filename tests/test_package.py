from smolhub.scripts.finetune import SFTTrainer
from smolhub.scripts.preference import PreferenceAlignmentTrainer
from smolhub.scripts.preference import PreferenceAlignmentTrainer
from smolhub.scripts.lora import LoRAModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from smolhub.helper.dataset.dataset_main import PreprocessDataset
from smolhub.helper.scheduler import CustomLRScheduler
# from load_config import Config
import torch
from smolhub.helper.dataset.load_config import Config

model_id = "openai-community/gpt2"

config = Config().get_config()
dataset_path = config["Dataset"]["dataset_path"]


sft_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct", token = config["huggingface"]["hf_token"])
ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct", token = config["huggingface"]["hf_token"])
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

# if tokenizer.pad_token is None:
#     # Set the pad token to the eos token if it doesn't exist
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#     # tokenizer.pad_token = tokenizer.eos_token
    

#     print("Setting pad token as PAD token ")

# model.resize_token_embeddings(len(tokenizer))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# lora_model = LoRAModel(model)

optimizer = torch.optim.Adam(sft_model.parameters(), lr=1e-6)
scheduler = CustomLRScheduler(optimizer, warmup_iters=100, lr_decay_iters=2000, min_lr=1e-6, max_lr=2e-3, _type="cosine")


#Loading the dataset
preprocess_dataset = PreprocessDataset(dataset_path=dataset_path, tokenizer=tokenizer, device=device)
train_dataloader, val_dataloader, test_dataloader = preprocess_dataset.prepare_dataset()

#Initialize the Trainer
pref_trainer = PreferenceAlignmentTrainer(device, ref_model, sft_model, train_dataloader, val_dataloader, test_dataloader, optimizer, None, tokenizer, scheduler)

#Train
pref_trainer.train()