#!/bin/bash

# Create necessary directories
mkdir -p generated_data
mkdir -p checkpoints

# Install required packages
pip install -r gradio/requirements.txt

echo "Setup complete! You can now run:"
echo "1. python trainer.py - to start training"
echo "2. python gradio/app.py - to run the Gradio interface"
echo "3. For distributed training: torchrun --nproc_per_node=2 trainer.py"
