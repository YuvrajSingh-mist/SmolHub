# SmolTransformer

A compact implementation of an Encoder-Decoder Transformer for sequence-to-sequence translation tasks. This project implements a translation model from English to Hindi using the Samanantar dataset.

## Features

- **Encoder-Decoder Architecture**: Full transformer implementation with separate encoder and decoder
- **Sinusoidal Positional Embeddings**: Learnable position encoding for better sequence understanding
- **Multi-Head Attention**: Self-attention and cross-attention mechanisms
- **Advanced Generation**: Top-K sampling and beam search for text generation
- **Mixed Precision Training**: Automatic mixed precision for faster training
- **Gradient Accumulation**: Support for large effective batch sizes
- **Comprehensive Logging**: WandB integration for experiment tracking

## Architecture

### Model Components

1. **Encoder**:
   - Multi-layer encoder blocks with self-attention
   - Sinusoidal positional embeddings
   - Layer normalization and feed-forward networks

2. **Decoder**:
   - Multi-layer decoder blocks with masked self-attention
   - Cross-attention to encoder outputs
   - Autoregressive generation capability

3. **Attention Mechanisms**:
   - Masked Multi-Head Attention (for decoder self-attention)
   - Full Multi-Head Attention (for encoder self-attention)
   - Cross Multi-Head Attention (for encoder-decoder attention)

## Installation

```bash
# Clone the repository
cd SmolTransformer

# Install dependencies
chmod +x install.sh
./install.sh
```

## Configuration

The model configuration can be modified in `config.py`:

```python
@dataclass
class ModelArgs:
    block_size: int = 512           # Maximum sequence length
    batch_size: int = 32            # Training batch size
    embeddings_dims: int = 512      # Model embedding dimensions
    no_of_heads: int = 8            # Number of attention heads
    no_of_decoder_layers: int = 6   # Number of decoder layers
    max_lr: float = 6e-4           # Maximum learning rate
    # ... additional parameters
```

## Usage

### Training

```bash
python trainer.py
```

### Web Application

Launch the interactive Gradio web interface:

```bash
python launch_app.py
```

The app will be available at `http://localhost:7860` and provides:
- **Interactive Translation**: English to Hindi translation interface
- **Multiple Generation Methods**: Top-K sampling and beam search
- **Real-time Parameter Adjustment**: Temperature, top-k, beam width controls
- **Model Loading**: Load trained checkpoints
- **Example Translations**: Pre-built examples to try

### Web App Features

- 🌐 **User-friendly Interface**: Clean, intuitive web UI
- 🔧 **Configurable Parameters**: Adjust generation settings in real-time
- 📊 **Model Information**: Display architecture and training details
- 🎯 **Multiple Methods**: Compare Top-K sampling vs Beam search
- 💾 **Checkpoint Loading**: Load your trained models
- 📝 **Examples**: Built-in examples to test the model

### Key Training Features

- **Gradient Accumulation**: Automatically calculated based on `total_batch_size`
- **Learning Rate Scheduling**: Warmup + cosine decay
- **Mixed Precision**: Automatic mixed precision training
- **Checkpointing**: Regular model checkpoints saved to `checkpoints/`
- **Generation Samples**: Periodic text generation during training

### Dataset

The model is trained on the Hindi-English Samanantar dataset:
- **Source**: English text
- **Target**: Hindi text  
- **Preprocessing**: Automatic tokenization with IndicBARTSS tokenizer

## File Structure

```
SmolTransformer/
├── config.py          # Model configuration and hyperparameters
├── model.py           # Transformer model implementation
├── data.py            # Dataset loading and preprocessing
├── tokenizer.py       # Tokenizer setup and utilities
├── trainer.py         # Training loop and utilities
├── inference.py       # Text generation functions
├── install.sh         # Installation script
├── README.md          # This file
├── checkpoints/       # Model checkpoints
├── generated_data/    # Generated text samples
├── gradio/            # Gradio interface (optional)
└── old/              # Backup files
```

## Model Parameters

- **Parameters**: ~25M (configurable)
- **Context Length**: 512 tokens
- **Vocabulary**: IndicBARTSS tokenizer (~30K tokens)
- **Architecture**: 6-layer encoder-decoder

## Training Features

### Optimization
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 6e-4 with warmup and cosine decay
- **Gradient Clipping**: 1.0 max norm
- **Mixed Precision**: Automatic FP16 training

### Monitoring
- **WandB Integration**: Comprehensive experiment tracking
- **Metrics**: Loss, perplexity, gradient norms
- **Generation Samples**: Regular text generation examples
- **Validation**: Periodic validation loss evaluation

### Generation Methods
- **Top-K Sampling**: Configurable top-k and temperature
- **Beam Search**: Multi-beam search with configurable width
- **Repetition Penalty**: Reduces repetitive generation

## Example Usage

```python
from model import Transformer
from tokenizer import initialize_tokenizer
from inference import topk_sampling, beam_search_corrected

# Initialize model and tokenizer
tokenizer = initialize_tokenizer()
model = Transformer(src_vocab_size=len(tokenizer), tgt_vocab_size=len(tokenizer))

# Generate text
prompt = "Hello, how are you?"
generated = topk_sampling(model, prompt, tokenizer, device="cuda", max_length=50)
print(generated)
```

## Customization

### Adding New Datasets
Modify `data.py` to load your dataset:

```python
def load_datasets(token, sample_size=None):
    # Load your custom dataset here
    dataset = load_dataset("your_dataset")
    return dataset
```

### Changing Model Size
Adjust parameters in `config.py`:

```python
embeddings_dims = 768  # Larger model
no_of_heads = 12       # More attention heads
no_of_decoder_layers = 12  # Deeper model
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers
- Datasets
- WandB
- CUDA-capable GPU (recommended)

## Model Resources

- **Hugging Face Model**: [YuvrajSingh9886/SmolTransformer](https://huggingface.co/YuvrajSingh9886/SmolTransformer)
- **Training Report**: [Weights & Biases Report](https://wandb.ai/rentio/Translation/reports/Translation--VmlldzoxMzY3OTg3MQ?accessToken=3hspzhfiyo1ekagen3o0ly0nmuqhhs5jzfpno9vb0oei2rwyum0hsgdrmfjqsycg)

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
