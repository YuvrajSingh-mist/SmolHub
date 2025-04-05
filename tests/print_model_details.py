
from torchinfo import summary

def print_model(model, dataloader):
    # lora_model = LoRAModel()
    model.to(ModelArgs.device)
    input_data = next(iter(dataloader))
    #Printing a summary of the architecture

    summary(model=model,
            input_data=input_data,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
