#!/bin/bash

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install datasets
pip install wandb
pip install liger-kernel
pip install tqdm
pip install torchinfo
pip install gradio

echo "Installation complete!"
