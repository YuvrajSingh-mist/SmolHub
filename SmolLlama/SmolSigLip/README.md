# SmolSigLip - Vision-Language Model with Wandb Integration

This project implements a compact SigLIP (Sigmoid Loss for Language Image Pre-training) model with comprehensive Weights & Biases (wandb) integration for experiment tracking and visualization.

## Features

- **Contrastive Vision-Language Learning**: Based on SigLIP architecture
- **Comprehensive Logging**: Detailed wandb integration for experiment tracking
- **Dynamic Dataset Loading**: Streaming support for large datasets like COYO-700M
- **Model Artifacts**: Automatic model saving and versioning through wandb
- **Real-time Visualizations**: Training curves, gradient tracking, and model summaries

## Setup

### 1. Install Dependencies

```bash
cd /speech/advait/yuvraj/LLMs/SmolSigLip
pip install -r requirements.txt
```

### 2. Login to Wandb

```bash
wandb login
```

Enter your API key when prompted. You can find it at https://wandb.ai/authorize

## Usage

### Basic Training with Wandb

```bash
python train.py
```

This will:
- Initialize a wandb run with project name "SmolSigLip"
- Log model architecture and parameters
- Track training/validation losses in real-time
- Save model artifacts
- Generate training visualizations

### Custom Training Script

```python
import torch
from engine import train, initialize_wandb, log_model_summary

# Your model and data setup
model = YourSigLipModel()
train_loader = DataLoader(...)
val_loader = DataLoader(...)

# Initialize wandb with custom config
config = {
    "learning_rate": 1e-4,
    "batch_size": 32,
    "epochs": 10,
    # ... other hyperparameters
}

initialize_wandb(
    project_name="SmolSigLip",
    config=config,
    run_name="custom_experiment"
)

# Log model summary
log_model_summary(model)

# Train with automatic wandb logging
results = train(
    model=model,
    train_dataloader=train_loader,
    test_dataloader=val_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=10,
    device=device
)
```

## Wandb Features

### 1. Real-time Metrics

The following metrics are automatically logged:

**Training Metrics:**
- `train/batch_loss`: Loss for each training batch
- `train/learning_rate`: Current learning rate
- `train/step`: Global training step
- `train/epoch`: Current epoch

**Validation Metrics:**
- `val/batch_loss`: Validation loss per batch
- `val/epoch`: Current epoch

**Epoch-level Metrics:**
- `epoch/train_loss`: Average training loss per epoch
- `epoch/test_loss`: Average validation loss per epoch
- `epoch/loss_ratio`: Test/Train loss ratio (overfitting indicator)

**Model Metrics:**
- `model/total_parameters`: Total model parameters
- `model/trainable_parameters`: Trainable parameters
- `model/architecture`: Model architecture visualization

**Gradient Tracking:**
- `gradients/total_norm`: Total gradient norm
- `gradients/param_count`: Number of parameters with gradients
- `gradients/{layer_name}`: Individual layer gradient norms

### 2. Model Artifacts

Models are automatically saved as wandb artifacts:
- Model state dictionaries
- Training checkpoints
- Final trained models

### 3. Custom Visualizations

Training plots are automatically generated and logged:
- Training vs Validation loss curves
- Overfitting monitoring (loss ratio plots)
- Learning rate schedules

## Project Structure

```
SmolSigLip/
├── engine.py              # Core training engine with wandb integration
├── train.py               # Main training script
├── train_with_wandb.py    # Example script with full wandb setup
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── test_*.py             # Various test scripts
```

## Configuration Options

### Wandb Configuration

```python
config = {
    # Model parameters
    "learning_rate": 1e-4,
    "batch_size": 32,
    "epochs": 10,
    
    # Architecture
    "image_size": 224,
    "text_max_length": 77,
    
    # Optimization
    "optimizer": "AdamW",
    "weight_decay": 0.01,
    "lr_scheduler": "cosine",
    
    # Dataset
    "dataset": "COYO-700M",
    "subset_size": 10000,
}
```

### Engine Functions

#### `initialize_wandb(project_name, config, run_name)`
Initializes a wandb run with proper configuration.

#### `log_model_summary(model, input_sample)`
Logs model architecture and parameter counts.

#### `log_gradients(model, step)`
Logs gradient statistics during training.

#### `train(model, train_dataloader, test_dataloader, ...)`
Main training function with automatic wandb logging.

## Monitoring Training

### Wandb Dashboard

After starting training, visit your wandb dashboard to monitor:

1. **Overview**: Key metrics summary
2. **Charts**: Real-time training curves
3. **System**: GPU/CPU utilization
4. **Logs**: Console output
5. **Artifacts**: Saved models
6. **Files**: Generated plots and logs

### Key Metrics to Watch

- **Loss Ratio**: Keep test/train loss ratio close to 1.0
- **Gradient Norm**: Monitor for exploding/vanishing gradients
- **Learning Rate**: Ensure proper LR scheduling
- **Parameter Count**: Verify model size

## Troubleshooting

### Common Issues

1. **Wandb Login Issues**
   ```bash
   wandb login --relogin
   ```

2. **Dataset Loading Errors**
   - The script automatically falls back to dummy data if COYO dataset fails
   - Check internet connection for streaming datasets

3. **Memory Issues**
   - Reduce batch_size in configuration
   - Use gradient checkpointing if available

4. **GPU Issues**
   - Ensure CUDA is properly installed
   - Check GPU memory with `nvidia-smi`

### Performance Tips

1. **Efficient Logging**
   - Validation metrics are logged every 10 batches to reduce overhead
   - Adjust logging frequency if needed

2. **Dataset Optimization**
   - Use appropriate num_workers for DataLoader
   - Consider dataset caching for repeated experiments

3. **Model Optimization**
   - Use mixed precision training
   - Enable torch.compile for PyTorch 2.0+

## Examples

### Quick Start

```bash
# Basic training with default settings
python train.py
```

### Custom Experiment

```bash
# Training with custom configuration
python train_with_wandb.py
```

### Advanced Usage

```python
# Custom training loop with detailed logging
import wandb
from engine import initialize_wandb, log_gradients

initialize_wandb("SmolSigLip", config)

for epoch in range(epochs):
    for step, batch in enumerate(train_loader):
        # ... training step ...
        
        # Log custom metrics
        wandb.log({
            "custom/metric": custom_value,
            "debug/tensor_norm": tensor.norm().item()
        }, step=global_step)
        
        # Log gradients every 100 steps
        if step % 100 == 0:
            log_gradients(model, global_step)
```

## Contributing

When adding new features:

1. Update wandb logging for new metrics
2. Add configuration options to the config dict
3. Update this README with new features
4. Test with both dummy and real datasets

## License

[Add your license information here]
