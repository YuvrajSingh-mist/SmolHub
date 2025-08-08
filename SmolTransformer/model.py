import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import ModelArgs
from tokenizer import get_tokenizer

# Initialize model args
model_args = ModelArgs()

# Global tokenizer - will be set later
tokenizer = None

def initialize_tokenizer_in_model(hf_token=None):
    """Initialize the global tokenizer with the provided HF token"""
    global tokenizer
    if tokenizer is None:
        from tokenizer import initialize_tokenizer
        tokenizer = initialize_tokenizer(hf_token)
    return tokenizer

class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(
        self,
        embeddings_dims=model_args.embeddings_dims,
        max_seq_len=model_args.block_size,
        theta=10000.0
    ):
        super().__init__()
        self.embeddings_dims = embeddings_dims
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Create the positional encoding matrix
        pe = torch.zeros(max_seq_len, embeddings_dims)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        # Calculate the div_term for the sinusoidal pattern
        div_term = torch.exp(torch.arange(0, embeddings_dims, 2).float() * 
                           -(math.log(theta) / embeddings_dims))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, embeddings_dims)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, embeddings_dims)
        seq_len = x.shape[1]
        # Access the registered buffer correctly
        pe = getattr(self, 'pe')
        return pe[:, :seq_len, :]

class TgtTextEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size,
        embeddings_dims=model_args.embeddings_dims,
        device=model_args.device
    ):
        super().__init__()
        self.embeddings_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeddings_dims, device=device)

    def forward(self, x):
        return self.embeddings_table(x)

class SrcTextEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size,
        embeddings_dims=model_args.embeddings_dims,
        device=model_args.device
    ):
        super().__init__()
        self.embeddings_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeddings_dims, device=device)

    def forward(self, x):
        return self.embeddings_table(x)

class LayerNormalization(nn.Module):
    def __init__(
        self,
        embeddings_dims=model_args.embeddings_dims
    ):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=embeddings_dims)
    
    def forward(self, x):
        return self.norm(x)

class MLPBlock(nn.Module):
    def __init__(
        self,
        dropout=model_args.dropout,
        embeddings_size=model_args.embeddings_dims,
        device=model_args.device
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(device=device, in_features=embeddings_size, out_features=4 * embeddings_size),
            nn.GELU(),
            nn.Linear(device=device, in_features=4 * embeddings_size, out_features=embeddings_size),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.mlp(x)

class MaskedAttentionHead(nn.Module):
    def __init__(
        self,
        attn_dropout=model_args.attn_dropout,
        embeddings_dims=model_args.embeddings_dims,
        no_of_heads=model_args.no_of_heads,
        device=model_args.device
    ):
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.keys = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.values = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.dropout = nn.Dropout(p=attn_dropout)

    def forward(self, x, mask=None):
        batch, block_size, embd_dims = x.shape
        
        k = self.keys(x)
        q = self.query(x)
        v = self.values(x)
        weights = q @ torch.transpose(k, dim0=-2, dim1=-1) * (k.shape[-1] ** -0.5)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            masked_values = weights.masked_fill(mask == 0, float('-inf'))
            masked_table = torch.tril(torch.ones(block_size, block_size, device=x.device))
            masked_values = masked_values.masked_fill(masked_table[:block_size, :block_size] == 0, float('-inf'))
            weights_normalized = nn.functional.softmax(masked_values, dim=-1)
            out = weights_normalized @ v
            return out
        else:
            masked_table = torch.tril(torch.ones(block_size, block_size, device=x.device))
            weights = weights.masked_fill(masked_table[:block_size, :block_size] == 0, float('-inf'))
            weights_normalized = nn.functional.softmax(weights, dim=-1)
            out = weights_normalized @ v
            return out

class MaskedMHA(nn.Module):
    def __init__(
        self,
        attn_dropout=model_args.attn_dropout,
        embeddings_dims=model_args.embeddings_dims,
        no_of_heads=model_args.no_of_heads,
        device=model_args.device
    ):
        super().__init__()
        self.heads = nn.ModuleList([MaskedAttentionHead(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads) for _ in range(no_of_heads)])
        self.dropout = nn.Dropout(p=attn_dropout)
        self.linear = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims, device=device, bias=False)

    def forward(self, x, mask=None):
        concat = torch.cat([head(x, mask) for head in self.heads], dim=-1)
        linear_layer = self.linear(concat)
        out = self.dropout(linear_layer)
        return out

class CrossAttentionHead(nn.Module):
    def __init__(
        self,
        attn_dropout=model_args.attn_dropout,
        embeddings_dims=model_args.embeddings_dims,
        no_of_heads=model_args.no_of_heads,
        device=model_args.device
    ):
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.keys = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.values = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.dropout = nn.Dropout(p=attn_dropout)

    def forward(self, k, v, q, srcmask=None):
        # Project inputs to queries, keys, values
        query = self.query(q)  # Query comes from decoder
        key = self.keys(k)     # Key comes from encoder
        value = self.values(v) # Value comes from encoder
        
        # Calculate attention scores
        weights = query @ torch.transpose(key, dim0=-2, dim1=-1) * (key.shape[-1] ** -0.5)

        if srcmask is not None:
            srcmask = srcmask.unsqueeze(1)
            masked_values = weights.masked_fill(srcmask == 0, float('-inf'))
            weights_normalized = nn.functional.softmax(masked_values, dim=-1)
            out = weights_normalized @ value
            return out
        else:
            weights_normalized = nn.functional.softmax(weights, dim=-1)
            out = weights_normalized @ value
            return out

class FullAttentionHead(nn.Module):
    def __init__(
        self,
        attn_dropout=model_args.attn_dropout,
        embeddings_dims=model_args.embeddings_dims,
        no_of_heads=model_args.no_of_heads,
        device=model_args.device
    ):
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.keys = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.values = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.dropout = nn.Dropout(p=attn_dropout)

    def forward(self, x, mask=None):
        k = self.keys(x)
        q = self.query(x)
        v = self.values(x)
        weights = q @ torch.transpose(k, dim0=-2, dim1=-1) * (k.shape[-1] ** -0.5)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            masked_values = weights.masked_fill(mask == 0, float('-inf'))
            weights_normalized = nn.functional.softmax(masked_values, dim=-1)
            out = weights_normalized @ v
            return out
        else:
            weights_normalized = nn.functional.softmax(weights, dim=-1)
            weights_normalized = self.dropout(weights_normalized)
            out = weights_normalized @ v
            out = self.dropout(out)
            return out

class FullMHA(nn.Module):
    def __init__(
        self,
        attn_dropout=model_args.attn_dropout,
        embeddings_dims=model_args.embeddings_dims,
        no_of_heads=model_args.no_of_heads,
        device=model_args.device
    ):
        super().__init__()
        self.heads = nn.ModuleList([FullAttentionHead(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads) for _ in range(no_of_heads)])
        self.dropout = nn.Dropout(p=attn_dropout)
        self.linear = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims, device=device, bias=False)

    def forward(self, x, mask=None):
        concat = torch.cat([head(x, mask) for head in self.heads], dim=-1)
        linear_layer = self.linear(concat)
        out = self.dropout(linear_layer)
        return out

class CrossMHA(nn.Module):
    def __init__(
        self,
        attn_dropout=model_args.attn_dropout,
        embeddings_dims=model_args.embeddings_dims,
        no_of_heads=model_args.no_of_heads,
        device=model_args.device
    ):
        super().__init__()
        self.heads = nn.ModuleList([CrossAttentionHead(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads) for _ in range(no_of_heads)])
        self.dropout = nn.Dropout(p=attn_dropout)
        self.linear = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims, device=device, bias=False)

    def forward(self, value, key, x, srcmask=None):
        concat = torch.cat([head(value, key, x, srcmask) for head in self.heads], dim=-1)
        linear_layer = self.linear(concat)
        out = self.dropout(linear_layer)
        return out

class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        attn_dropout=model_args.attn_dropout,
        embeddings_dims=model_args.embeddings_dims,
        no_of_heads=model_args.no_of_heads,
        dropout=model_args.dropout,
    ):
        super().__init__()
        self.cross = CrossMHA(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads)
        self.masked = MaskedMHA(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads)
        self.layer_norm1 = LayerNormalization(embeddings_dims)
        self.layer_norm2 = LayerNormalization(embeddings_dims)
        self.layer_norm4 = LayerNormalization(embeddings_dims)
        self.mlp_block = MLPBlock(dropout=dropout, embeddings_size=embeddings_dims)

    def forward(self, key, value, x, Srcmask=None, Targetmask=None):
        x = self.layer_norm1(x + self.masked(x, Targetmask))
        x = self.layer_norm2(x + self.cross(key, value, x, Srcmask))
        x = self.layer_norm4(x + self.mlp_block(x))
        return x

class DecoderModel(nn.Module):
    def __init__(
        self,
        tgt_vocab_size,
        attn_dropout=model_args.attn_dropout,
        embeddings_dims=model_args.embeddings_dims,
        no_of_heads=model_args.no_of_heads,
        block_size=model_args.block_size,
        dropout=model_args.dropout,
        no_of_decoder_layers=model_args.no_of_decoder_layers,
    ):
        super().__init__()
        self.tgt_text_embds = TgtTextEmbeddings(vocab_size=tgt_vocab_size, embeddings_dims=embeddings_dims)
        self.decoder_layers = nn.ModuleList([TransformerDecoderBlock(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads, dropout=dropout) for _ in range(no_of_decoder_layers)])
        self.apply(self._init_weights)
        
        self.positional_embeddings_tgt = SinusoidalPositionalEmbeddings(
            embeddings_dims=embeddings_dims, 
            max_seq_len=block_size, 
            theta=10000.0
        )
        self.dropout = nn.Dropout(p=dropout)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, key, value, x, srcmask=None, target_mask=None):
        x = self.tgt_text_embds(x)
        x = self.dropout(x)
        x = x + self.positional_embeddings_tgt(x)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(key, value, x, srcmask, target_mask)
        x = self.dropout(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        attn_dropout=model_args.attn_dropout,
        embeddings_dims=model_args.embeddings_dims,
        no_of_heads=model_args.no_of_heads,
        dropout=model_args.dropout,
    ):
        super().__init__()
        self.mha = FullMHA(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads)
        self.layer_norm1 = LayerNormalization(embeddings_dims)
        self.layer_norm2 = LayerNormalization(embeddings_dims)
        self.mlp_block = MLPBlock(dropout=dropout, embeddings_size=embeddings_dims)

    def forward(self, x, mask=None):
        x = self.layer_norm1(x + self.mha(x, mask))
        x = self.layer_norm2(x + self.mlp_block(x))
        return x

class EncoderModel(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        attn_dropout=model_args.attn_dropout,
        embeddings_dims=model_args.embeddings_dims,
        no_of_heads=model_args.no_of_heads,
        block_size=model_args.block_size,
        dropout=model_args.dropout,
        no_of_decoder_layers=model_args.no_of_decoder_layers,
    ):
        super().__init__()
        
        self.positional_embeddings_src = SinusoidalPositionalEmbeddings(
            embeddings_dims=embeddings_dims, 
            max_seq_len=block_size, 
            theta=10000.0
        )

        self.src_text_embeds = SrcTextEmbeddings(vocab_size=src_vocab_size, embeddings_dims=embeddings_dims)
        self.encoder_layers = nn.ModuleList([TransformerEncoderBlock(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads, dropout=dropout) for _ in range(no_of_decoder_layers)])
        self.apply(self._init_weights)
        self.dropout = nn.Dropout(p=dropout)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, mask):
        x = self.src_text_embeds(x)
        x = x + self.positional_embeddings_src(x)
        x = self.dropout(x)
        
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        x = self.dropout(x)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, use_liger=True):
        super().__init__()
        self.encoder = EncoderModel(src_vocab_size=src_vocab_size)
        self.decoder = DecoderModel(tgt_vocab_size=tgt_vocab_size)
        self.norm = LayerNormalization(embeddings_dims=model_args.embeddings_dims)
        self.linear_layer = nn.Linear(in_features=model_args.embeddings_dims, out_features=tgt_vocab_size, device=model_args.device, bias=False)
        
        self.use_liger = use_liger
        if use_liger:
            try:
                from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
                self.le_loss = LigerFusedLinearCrossEntropyLoss(
                    ignore_index=get_tokenizer().pad_token_id if get_tokenizer() else 0
                )
            except ImportError:
                print("Liger kernel not available, using standard cross entropy")
                self.use_liger = False
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, src, tgt_idx, tgt, src_mask=None, tgt_mask=None, inference=False):
        x = self.encoder(src, src_mask)
        x = self.decoder(x, x, tgt_idx, src_mask, tgt_mask)
        x = self.norm(x)
        
        if inference:
            out = self.linear_layer(x)
            return out
            
        if self.use_liger:
            y = x.contiguous().view(-1, model_args.embeddings_dims)
            if tgt is not None:
                labels = tgt.contiguous().view(-1)
                loss = self.le_loss(self.linear_layer.weight, y, labels)
                return loss
        else:
            # Standard cross-entropy loss calculation
            logits = self.linear_layer(x)
            logits = logits.view(-1, logits.size(-1))
            targets = tgt.contiguous().view(-1)
            
            # Get pad token id from tokenizer
            pad_token_id = get_tokenizer().pad_token_id if get_tokenizer() else 0
            loss = F.cross_entropy(logits, targets, ignore_index=pad_token_id, label_smoothing=0.1)
            return loss
