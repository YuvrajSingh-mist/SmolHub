#!/bin/bash

# SmolTransformer Installation Script

echo "Installing SmolTransformer dependencies..."

# Update pip
pip install --upgrade pip

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers and datasets
pip install transformers datasets

# Install training utilities
pip install wandb tqdm

# Install Gradio for the web app
pip install gradio

# Install optional optimization libraries
pip install liger-kernel || echo "Liger kernel installation failed, continuing without it"

# Install additional utilities
pip install torchinfo

echo "Installation complete!"
echo ""
echo "🚀 Usage:"
echo "  Training: python trainer.py"
echo "  Web App:  python launch_app.py"
echo ""
echo "📁 The web app will be available at http://localhost:7860"
