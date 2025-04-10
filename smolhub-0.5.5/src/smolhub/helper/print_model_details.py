import torch
from torchinfo import summary

from smolhub.scripts.lora import ModelArgs

def print_model(model, dataloader):
    # lora_model = LoRAModel()
    model.to(ModelArgs.device)
    # input_data = next(iter(dataloader))
    input_ids = torch.randint(
    0, model.config.vocab_size,
    (1, model.config.n_ctx)
).to(ModelArgs.device)
    #Printing a summary of the architecture
    
    summary(model=model,
            input_data=input_ids,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
