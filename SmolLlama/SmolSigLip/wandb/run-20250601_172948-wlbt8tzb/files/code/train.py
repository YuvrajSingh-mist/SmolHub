import torch
import wandb
from dataclasses import dataclass
from transformers import RobertaTokenizer, RobertaModel

from torch.utils.data import Dataset, DataLoader

from torch.utils.data import random_split
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoImageProcessor, AutoModelForImageClassification

import pandas as pd
import timm
import torch.nn as nn
import requests
from io import BytesIO
import datasets
from datasets import load_dataset
import numpy as np
import random


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
    projection_dims = 768
    attn_dropout = 0.1
    no_of_heads = 12 #IMP needs to be thoroughly calculated
    dropout = 0.1
    epochs = 10000 * 7
    lr = 4e-4
    no_of_decoder_layers = 12 #IMP needs to be thoroughly calculated
    weight_decay_optim = 0.2
    beta_1 = 0.9
    beta_2 = 0.95
    epsilon = 1e-6
    device = 'cuda:3'
    vocab_size = 2000
    head_lr = 1e-3
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
        self.multimodalTextLayerProjector = nn.Linear(in_features=ModelArgs.text_embeddings_dims, out_features=ModelArgs.projection_dims, device=ModelArgs.device)

        for p in self.model.parameters():
            p.requires_grad = True
        self.model.train()
        
        
    def forward(self, x):
        # print("Problemetic x shape: ", x['input_ids'].shape)
        # print("Problemetic x shape: ", x['attention_mask'].shape)
        x['input_ids'] = x['input_ids'].squeeze(1)
        x['attention_mask'] = x['attention_mask'].squeeze(1) 
        x = self.model(input_ids = x['input_ids'], attention_mask = x['attention_mask'])['last_hidden_state'][:, 0, :] 
        # print(x)
        x = self.layer_norm(x)
        return self.multimodalTextLayerProjector(x)
    
    
class VisionModel(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=ModelArgs.model_name, pretrained=ModelArgs.pretrained, trainable=ModelArgs.trainable
    ):
        super().__init__()
        self.model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
        self.preprocessor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.model.train()

    def forward(self, x):
        # Preprocess the image
        inputs = self.preprocessor(images=x['image'], return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.pooler_output
        return logits
    
    
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
siglip = siglip.to(ModelArgs.device)

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
params = [
        {"params": siglip.vision.parameters(), "lr": ModelArgs.image_encoder_lr},
        {"params": siglip.text.parameters(), "lr": ModelArgs.text_encoder_lr},
        {"params": itertools.chain(
            siglip.multimodalVisionLayerProjector.parameters(), siglip.multimodelTextLayerPorjector.parameters(), [siglip.temperature]
        ), "lr": ModelArgs.head_lr, "weight_decay": ModelArgs.weight_decay_optim}
    ]

optimizer = torch.optim.Adam(lr=ModelArgs.lr, params=params, eps=ModelArgs.epsilon)

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
        "batch_size": ModelArgs.batch_size,
        "epochs": ModelArgs.epochs,
        "model_name": "SigLip",
        "optimizer": "Adam",
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
    # Log model summary
    siglip = torch.compile(siglip)
    
    
    results = engine.train(model=siglip,
                           writer=None,
                           train_dataloader=trainloader,
                           test_dataloader=valloader,
                           optimizer=optimizer,
                           gradient_accumulation_steps= (ModelArgs.total_batch_size // ModelArgs.batch_size),
                           loss_fn=loss_fn,
                           epochs=ModelArgs.epochs,
                           device=torch.device(ModelArgs.device),
                           validate_every_n_steps=500)
    
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

