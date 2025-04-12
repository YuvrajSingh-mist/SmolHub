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
  author: Yuvraj Singh
  version: 1.0

LoRA:
  rank: 4
  alpha: 8

Preference:
  beta: 0.1
  
Dataset:
    use_hf_dataset: True
    dataset_path: trl-lib/ultrafeedback_binarized
    max_length: 512
    batch_size:  16
    num_workers:  4
    shuffle:  True
    drop_last: True
    pin_memory:  True
    persistent_workers:  True
    type: "classification" #TODO Add Chat style and Instruction 
   
huggingface:
  hf_token: "..."

Training: 
  type: 'preference'

Model:
 
  epochs: 1
  eval_iters: 10
  eval_steps: 0
  save_model_path: "saved_model"
  saved_model_name: 'model.pt'


MAP:
  use_bfloat16:  False
  use_float16: False

Optimizations:
  use_compile: False


wandb:
  project_name: "SFTrainer"
  
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

