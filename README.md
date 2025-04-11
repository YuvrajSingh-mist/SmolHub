# SmolHub

A lightweight and efficient package for training language models using Low-Rank Adaptation (LoRA). Designed for easy experimentation and research with minimal boilerplate.

## Features

- 🚀 **Multiple Training Paradigms**:
  - Supervised Fine-tuning (SFT)
  - Pretraining
  - Preference Alignment (RLHF-style training)
- 📦 **Easy Integration** with Hugging Face models
- ⚡ **Efficient Training** with LoRA
- 📊 **WandB Integration** for experiment tracking
- 🔄 **Automatic Dataset Handling**

## Installation

```bash
pip install smolhub
```

## Quickstart

```python
from smolhub.scripts.finetune import SFTTrainer
from smolhub.scripts.lora import LoRAModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_id = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Setup LoRA
lora_model = LoRAModel(model)

# Train with minimal setup
trainer = SFTTrainer(
    model=lora_model,
    dataset_path="your_dataset",  # HF dataset or local file
    tokenizer=tokenizer
)
trainer.train()
```

## Configuration

SmolHub uses a YAML configuration file for experiment settings. A default config is created in your project directory:

```yaml
project:
  name: SFTrainer
  version: 1.0

LoRA:
  rank: 4    # LoRA rank for weight updates
  alpha: 8   # LoRA alpha parameter

Dataset:
  use_hf_dataset: true
  dataset_path: "MMEX/text-classification-dataset"
  type: "classification"  # Options: classification, pretraining, preference
  batch_size: 16
  max_length: 512

Model:
  epochs: 1
  eval_frequency: 100
  save_path: "saved_model"

# ... See documentation for full config options
```

## Training Modes

### Supervised Fine-tuning
```python
trainer = SFTTrainer(model, dataset_path="classification_dataset")
```

### Pretraining
```python
from smolhub.scripts.pretrain import PreTrainer
trainer = PreTrainer(model, dataset_path="text_corpus")
```

### Preference Alignment
```python
from smolhub.scripts.align import PreferenceTrainer
trainer = PreferenceTrainer(model, dataset_path="preference_pairs")
```

## Documentation

For detailed documentation and examples, visit our [documentation](https://github.com/yourusername/smolhub/wiki).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

