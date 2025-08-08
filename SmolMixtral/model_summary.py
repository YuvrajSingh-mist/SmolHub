import torch
from config import create_model_args
from model import Mixtral
from torchinfo import summary

# Create model args with tokenizer
model_args = create_model_args()

# Create model instance
model = Mixtral(
    attn_dropout=model_args.attn_dropout, 
    embeddings_dims=model_args.embeddings_dims, 
    no_of_heads=model_args.no_of_heads, 
    block_size=model_args.block_size, 
    dropout=model_args.dropout, 
    no_of_decoder_layers=model_args.no_of_decoder_layers, 
    vocab_size=model_args.vocab_size, 
    device=model_args.device,
    tokenizer=model_args.tokenizer
)
model = model.to(model_args.device)

# Create sample input
idx = torch.randint(
    low=0,
    high=model_args.vocab_size,
    size=(model_args.batch_size, model_args.block_size),
    dtype=torch.long
)
idx = idx.to(model_args.device)

# Print model summary
print(summary(
    model=model,
    input_data=idx,
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"]
))
