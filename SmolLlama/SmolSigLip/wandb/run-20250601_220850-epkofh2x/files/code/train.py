import torch
import wandb
from dataclasses import dataclass
from transformers import RobertaTokenizer, RobertaModel

from torch.utils.data import Dataset, DataLoader

from torch.utils.data import random_split
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoImageProcessor
from transformers import ViTModel
import pandas as pd
import timm
import torch.nn as nn
import requests
from io import BytesIO
import datasets
from datasets import load_dataset
import numpy as np
import random
import math


class LinearWarmupCosineDecayLR:
    """
    Custom learning rate scheduler with linear warmup and cosine decay.
    
    - Linear warmup from initial_lr to peak_lr for warmup_examples
    - Cosine decay from peak_lr to min_lr for remaining examples
    """
    def __init__(self, optimizer, initial_lr, peak_lr, min_lr, warmup_examples, total_examples, 
                 effective_batch_size, last_step=-1):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.effective_batch_size = effective_batch_size
        
        # Calculate steps from examples
        self.warmup_steps = warmup_examples // effective_batch_size
        self.total_steps = total_examples // effective_batch_size
        self.decay_steps = self.total_steps - self.warmup_steps
        
        self.step_count = last_step + 1
        
        print(f"📊 Learning Rate Scheduler Configuration:")
        print(f"   Initial LR: {initial_lr:.2e}")
        print(f"   Peak LR: {peak_lr:.2e}")
        print(f"   Min LR: {min_lr:.2e}")
        print(f"   Warmup examples: {warmup_examples:,}")
        print(f"   Total examples: {total_examples:,}")
        print(f"   Effective batch size: {effective_batch_size:,}")
        print(f"   Warmup steps: {self.warmup_steps:,}")
        print(f"   Total steps: {self.total_steps:,}")
        print(f"   Decay steps: {self.decay_steps:,}")
        
    def get_lr(self):
        """Calculate learning rate for current step."""
        if self.step_count < self.warmup_steps:
            # Linear warmup phase
            progress = self.step_count / self.warmup_steps
            lr = self.initial_lr + (self.peak_lr - self.initial_lr) * progress
        elif self.step_count < self.total_steps:
            # Cosine decay phase
            decay_progress = (self.step_count - self.warmup_steps) / self.decay_steps
            lr = self.min_lr + (self.peak_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * decay_progress))
        else:
            # After total steps, keep at minimum
            lr = self.min_lr
            
        return lr
    
    def step(self):
        """Update learning rate for all parameter groups."""
        lr = self.get_lr()
        
        # Update all parameter groups
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.step_count += 1
        return lr
    
    def get_last_lr(self):
        """Return the last computed learning rate."""
        return [self.get_lr()]


class COYODataset(Dataset):
    """
    COYO-700M dataset loader with streaming support and dynamic subset fetching
    """
    def __init__(self, 
                 split="train", 
                 subset_size=8192,
                 transform=None,
                 tokenizer=None,
                 max_length=256):
        
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.subset_size = subset_size
        self.current_offset = 0
        self.data_list = []
        self.dataset_stream = None
        self.exhausted = False
        
        print(f"Initializing COYO dataset with subset size {subset_size}...")
        
        # Try to load COYO dataset, fallback to dummy if not available
        try:
            print("Attempting to load COYO-700M dataset...")
            # Load COYO dataset from Hugging Face
            dataset = load_dataset("kakaobrain/coyo-700m", split=split, streaming=True)
            self.dataset_stream = iter(dataset.skip(self.current_offset))
            print("Successfully connected to COYO-700M dataset stream")
            self._load_next_subset()
                    
        except Exception as e:
            print(f"Could not load COYO dataset: {e}")
            print("This might be due to:")
            print("1. Network connectivity issues")
            print("2. Hugging Face authentication required")
            print("3. Dataset access restrictions")
            print("Creating dummy dataset for testing...")
            self.data_list = self._create_dummy_dataset(subset_size)
            self.exhausted = True  # Don't try to fetch more dummy data
    
    def _load_next_subset(self):
        """Load the next subset of data from the stream"""
        if self.exhausted or self.dataset_stream is None:
            return
            
        print(f"Loading subset starting from offset {self.current_offset}...")
        new_data = []
        
        try:
            for i, item in enumerate(self.dataset_stream):
                if i >= self.subset_size:
                    break
                
                # COYO dataset has 'url', 'text', 'id', 'width', 'height', etc.
                # Filter out items with missing or invalid data
                if 'url' in item and 'text' in item and item['url'] and item['text']:
                    # Only include items with reasonable image dimensions and text
                    # if (item.get('width', 0) >= 224 and 
                    #     item.get('height', 0) >= 224):
                    new_data.append(item)
                        
                if (i + 1) % 1000 == 0:
                    print(f"Processed {i+1}/{self.subset_size} samples, kept {len(new_data)} valid samples...")
                    
            if len(new_data) == 0:
                print("⚠️  No valid samples found in this batch, using dummy data...")
                self.data_list = self._create_dummy_dataset(self.subset_size)
                self.exhausted = True
                return
                    
            if len(new_data) < self.subset_size // 2:  # If we got less than half the requested samples
                print(f"⚠️  Stream may be exhausted! Only loaded {len(new_data)} valid samples.")
                self.exhausted = True
            else:
                print(f"Successfully loaded {len(new_data)} valid samples from COYO dataset")
                
            # Replace current data with new subset
            self.data_list = new_data
            self.current_offset += self.subset_size  # Increment by requested size, not actual loaded
            
        except Exception as e:
            print(f"Error loading subset: {e}")
            if len(self.data_list) == 0:  # If no data loaded yet, use dummy
                self.data_list = self._create_dummy_dataset(self.subset_size)
            self.exhausted = True
    
    def refresh_subset(self):
        """Manually refresh the dataset with a new subset"""
        if not self.exhausted:
            print("🔄 Refreshing dataset with new subset...")
            self._load_next_subset()
        else:
            print("⚠️  Dataset stream is exhausted, cannot fetch new subset")
    
    def _create_dummy_dataset(self, size):
        """Create a dummy dataset for testing when COYO is not available"""
        print(f"Creating {size} dummy samples...")
        dummy_data = []
        sample_texts = [
            "A beautiful sunset over the mountains",
            "A cat sitting on a windowsill", 
            "A red car parked on the street",
            "Children playing in a park",
            "A delicious pizza with multiple toppings",
            "A modern building with glass windows",
            "A forest path in autumn",
            "A beach with crystal clear water",
            "A dog running in a field",
            "A flower garden in full bloom"
        ]
        
        for i in range(size):
            text_idx = i % len(sample_texts)
            dummy_data.append({
                'id': i + self.current_offset,
                'url': f'https://via.placeholder.com/224x224/FF0000/FFFFFF?text=Sample+{i + self.current_offset}',
                'text': sample_texts[text_idx] + f" (sample {i + self.current_offset})",
                'width': 224,
                'height': 224,
                'clip_similarity_vitb32': 0.5,
                'aesthetic_score_laion_v2': 5.0
            })
        return dummy_data
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        try:
            item = self.data_list[idx]
            
            # Get image from URL or create dummy
            image = self._load_image_from_url(item['url'])
            
            # Clean and process text
            text = item['text'].strip()
            if len(text) == 0:
                text = "A placeholder image"
            
            # Apply image transforms
            if self.transform:
                # Convert PIL image to numpy array for albumentations
                image_array = np.array(image)
                transformed = self.transform(image=image_array)
                image = transformed['image']
            
            # Tokenize text
            if self.tokenizer:
                text_tokens = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
            else:
                text_tokens = text
            
            return {
                'image': image,
                'text': text_tokens,
                'raw_text': text
            }
            
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            return self._get_dummy_item()
    
    def _load_image_from_url(self, url, timeout=10, max_retries=3):
        """Load image from URL with retry logic"""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert('RGB')
                return image
            except Exception as e:
                if attempt == max_retries - 1:
                    # Return a dummy colored image
                    color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                    return Image.new('RGB', (224, 224), color=color)
                continue
    
    def _get_dummy_item(self):
        """Return a dummy item in case of errors"""
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        dummy_image = Image.new('RGB', (224, 224), color=color)
        dummy_text = "A placeholder image"
        
        if self.transform:
            image_array = np.array(dummy_image)
            transformed = self.transform(image=image_array)
            dummy_image = transformed['image']
        
        if self.tokenizer:
            text_tokens = self.tokenizer(
                dummy_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        else:
            text_tokens = dummy_text
        
        return {
            'image': dummy_image,
            'text': text_tokens,
            'raw_text': dummy_text
        }


@dataclass
class ModelArgs:
    #Hyperparameters
    img_size = (224, 224)
    block_size = 256
    batch_size = 128
    text_embeddings_dims = 768
    img_embeddings_dims = 768
    projection_dims = 768 * 2
    attn_dropout = 0.1
    no_of_heads = 12 #IMP needs to be thoroughly calculated
    dropout = 0.1
    epochs = 10000 * 7
    lr = 4e-4
    no_of_decoder_layers = 12 #IMP needs to be thoroughly calculated
    weight_decay_optim = 0.0001
    beta_1 = 0.9
    beta_2 = 0.95
    epsilon = 1e-6
    device = 'cuda:3'
    vocab_size = 2000
    head_lr = 1e-4
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    model_name = 'google/vit-base-patch16-224'
    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    bias = -10.0
    temperature = torch.log(torch.tensor(10.0))
    total_batch_size = 16384
    
class Normalization(torch.nn.Module):
    def __init__(
        self,
        embeddings_dims
    ):  
        super().__init__()
        self.layernorm_layer = torch.nn.LayerNorm(normalized_shape=embeddings_dims)
        
        
    def forward(self, x):
        
        x = self.layernorm_layer(x)
        return x
        
        
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')


class TextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
            
        
        
        self.layer_norm = Normalization(ModelArgs.text_embeddings_dims)
        self.model = RobertaModel.from_pretrained('roberta-base')
        self.tokenizer = tokenizer
        # self.multimodalTextLayerProjector = nn.Linear(in_features=ModelArgs.text_embeddings_dims, out_features=ModelArgs.projection_dims, device=ModelArgs.device)

        for p in self.model.parameters():
            p.requires_grad = True
        self.model.train()
        
        
    def forward(self, x):
        # print("Problemetic x shape: ", x['input_ids'].shape)
        # print("Problemetic x shape: ", x['attention_mask'].shape)
        x['input_ids'] = x['input_ids'].squeeze(1)
        x['attention_mask'] = x['attention_mask'].squeeze(1)
        
        # Ensure input tensors are on the same device as the model
        device = next(self.model.parameters()).device
        x['input_ids'] = x['input_ids'].to(device)
        x['attention_mask'] = x['attention_mask'].to(device)
        
        x = self.model(input_ids = x['input_ids'], attention_mask = x['attention_mask'])['last_hidden_state'][:, 0, :] 
        # print(x)
        # x = self.layer_norm(x)
        return x
    
    
class VisionModel(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=ModelArgs.model_name, pretrained=ModelArgs.pretrained, trainable=ModelArgs.trainable
    ):
        super().__init__()
        # Use ViT model with feature extraction capability, not a classification model
       
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.preprocessor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.model.train()

    def forward(self, x):
        # Preprocess the image
        inputs = self.preprocessor(images=x, return_tensors="pt")
        
        # Move all input tensors to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        
        # For ViTModel, get the CLS token representation which is the first token
        # of the last hidden state
        return outputs.last_hidden_state[:, 0, :]
    
    
class SigLip(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.vision = VisionModel()
        self.text = TextModel()
        # self.tokenizer = tokenizer
        self.multimodelTextLayerPorjector = nn.Linear(in_features=ModelArgs.text_embeddings_dims, out_features=ModelArgs.projection_dims, device=ModelArgs.device)
        self.multimodalVisionLayerProjector = nn.Linear(in_features=ModelArgs.img_embeddings_dims , out_features=ModelArgs.projection_dims, device=ModelArgs.device)
        # self.temperature = nn.Parameter(torch.ones(size=(ModelArgs.batch_size,), device=ModelArgs.device), requires_grad=True)
        self.temperature = nn.Parameter(ModelArgs.temperature, requires_grad=True)
        self.bias = nn.Parameter(torch.tensor(ModelArgs.bias, dtype=torch.float32), requires_grad=True)

    def forward(self, batch):
        
        embeds_text = self.text(batch['text'])
        # print("Inside CLiP text: ", embeds_text.shape)
        proj_txt = torch.nn.functional.normalize(self.multimodelTextLayerPorjector(embeds_text))
        embeds_img = self.vision(batch['image'])
        # print("Inside ViT: ", embeds_img.shape)
        proj_img = torch.nn.functional.normalize(self.multimodalVisionLayerProjector(embeds_img))
        # print(proj_txt.shape)
        # print(proj_img.shape)
        logits = -(proj_txt @ proj_img.T) * torch.exp(self.temperature) + self.bias
        # print("Inside CLiP logits shape: ", logits.shape)
        return logits
    
    
siglip = SigLip()
# Ensure all components are on the correct device
siglip = siglip.to(ModelArgs.device)

# Explicitly move all sub-models to device to ensure consistency
siglip.vision.model = siglip.vision.model.to(ModelArgs.device)
siglip.text.model = siglip.text.model.to(ModelArgs.device)

train_transforms = A.Compose(
    [   
        A.Resize(height=224, width=224),
        A.CenterCrop(height=224, width=224),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

test_tyransforms = A.Compose(
    [
        A.Resize(height=224, width=224),
        A.CenterCrop(height=224, width=224),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)



#Creating dataloaders

# Initialize tokenizer for text processing
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Create COYO datasets
print("Creating COYO train dataset...")
train_dataset = COYODataset(
    split="train",
    subset_size=8192,  # Updated to 8192 samples
    transform=train_transforms,
    tokenizer=tokenizer,
    max_length=ModelArgs.block_size
)

print("Creating COYO validation dataset...")
val_dataset = COYODataset(
    split="train",  # COYO doesn't have separate validation split
    subset_size=1024,  # Updated validation set
    transform=test_tyransforms,
    tokenizer=tokenizer,
    max_length=ModelArgs.block_size
)

trainloader = DataLoader(train_dataset, batch_size=ModelArgs.batch_size, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=ModelArgs.batch_size, shuffle=False)


import itertools

# Learning rate scheduler configuration
WARMUP_EXAMPLES = 200_000_000  # 200M examples for warmup
TOTAL_EXAMPLES = 700_000_000   # 700M total examples in dataset
EFFECTIVE_BATCH_SIZE = ModelArgs.total_batch_size  # 16384
PEAK_LR = 0.001  # Peak learning rate
MIN_LR = 0.0     # Final learning rate

params = [
        {"params": siglip.vision.parameters(), "lr": ModelArgs.image_encoder_lr},
        {"params": siglip.text.parameters(), "lr": ModelArgs.text_encoder_lr},
        {"params": itertools.chain(
            siglip.multimodalVisionLayerProjector.parameters(), siglip.multimodelTextLayerPorjector.parameters(), [siglip.temperature]
        ), "lr": ModelArgs.head_lr, "weight_decay": ModelArgs.weight_decay_optim}
    ]

optimizer = torch.optim.Adam(lr=ModelArgs.head_lr, params=params, eps=ModelArgs.epsilon)

# Create custom learning rate scheduler
lr_scheduler = LinearWarmupCosineDecayLR(
    optimizer=optimizer,
    initial_lr=ModelArgs.head_lr,  # Start from head learning rate (1e-4)
    peak_lr=PEAK_LR,               # Peak at 0.001
    min_lr=MIN_LR,                 # Decay to 0
    warmup_examples=WARMUP_EXAMPLES,
    total_examples=TOTAL_EXAMPLES,
    effective_batch_size=EFFECTIVE_BATCH_SIZE
)

loss_fn = nn.CrossEntropyLoss()

torch.set_float32_matmul_precision('high')

# Only run training if this script is executed directly
if __name__ == "__main__":
    import engine
    import wandb
    
    # Configuration for wandb
    config = {
        "learning_rate": ModelArgs.lr,
        "image_encoder_lr": ModelArgs.image_encoder_lr,
        "text_encoder_lr": ModelArgs.text_encoder_lr,
        "head_lr": ModelArgs.head_lr,
        "peak_lr": PEAK_LR,
        "min_lr": MIN_LR,
        "warmup_examples": WARMUP_EXAMPLES,
        "total_examples": TOTAL_EXAMPLES,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "warmup_steps": WARMUP_EXAMPLES // EFFECTIVE_BATCH_SIZE,
        "total_steps": TOTAL_EXAMPLES // EFFECTIVE_BATCH_SIZE,
        "batch_size": ModelArgs.batch_size,
        "epochs": ModelArgs.epochs,
        "model_name": "SigLip",
        "optimizer": "Adam",
        "lr_scheduler": "LinearWarmupCosineDecay",
        "weight_decay": ModelArgs.weight_decay_optim,
        "epsilon": ModelArgs.epsilon,
        "device": ModelArgs.device,
        "block_size": ModelArgs.block_size,
        "train_subset_size": 8192,
        "val_subset_size": 1024,
        "image_size": ModelArgs.img_size,
        "text_max_length": ModelArgs.block_size
    }
    
    # Initialize wandb
    engine.initialize_wandb(
        project_name="SmolSigLip",
        config=config,
        run_name=f"siglip_lr{ModelArgs.lr}_bs{ModelArgs.batch_size}_epochs70000"
    )
    
    engine.log_model_summary(siglip)
    
    # Verify all model components are on the correct device before compilation
    print(f"🔍 Device check before compilation:")
    print(f"   - Main model device: {next(siglip.parameters()).device}")
    print(f"   - Vision model device: {next(siglip.vision.model.parameters()).device}")
    print(f"   - Text model device: {next(siglip.text.model.parameters()).device}")
    
    # Compile the model with error handling
    try:
        print("🔧 Compiling model with torch.compile...")
        siglip = torch.compile(siglip)
        print("✅ Model compilation successful!")
    except Exception as e:
        print(f"⚠️  torch.compile failed: {e}")
        print("🔄 Continuing without compilation...")
        # Continue without compilation if it fails
    
    
    results = engine.train(model=siglip,
                           writer=None,
                           train_dataloader=trainloader,
                           test_dataloader=valloader,
                           optimizer=optimizer,
                           gradient_accumulation_steps= (ModelArgs.total_batch_size // ModelArgs.batch_size),
                           loss_fn=loss_fn,
                           epochs=ModelArgs.epochs,
                           device=torch.device(ModelArgs.device),
                           validate_every_n_steps=500,
                           lr_scheduler=lr_scheduler)
    
    # Log final results
    wandb.log({
        "final/best_train_loss": min(results["train_loss"]),
        "final/best_test_loss": min(results["test_loss"]),
        "final/final_train_loss": results["train_loss"][-1],
        "final/final_test_loss": results["test_loss"][-1],
        "final/epochs_completed": ModelArgs.epochs,
    })
    
    # Save model artifact
    run_id = wandb.run.id if wandb.run else "unknown"
    model_artifact = wandb.Artifact(
        name=f"siglip_model_{run_id}",
        type="model",
        description="Trained SigLip model on COYO dataset"
    )
    
    torch.save(siglip.state_dict(), "siglip_final.pt")
    model_artifact.add_file("siglip_final.pt")
    wandb.log_artifact(model_artifact)
    
    # Finish wandb run
    wandb.finish()
    
    print("Training completed! Check your wandb dashboard for visualizations.")
    print(f"Final train loss: {results['train_loss'][-1]:.4f}")
    print(f"Final test loss: {results['test_loss'][-1]:.4f}")

