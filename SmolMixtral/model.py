import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint

from liger_kernel.transformers import LigerLayerNorm
from liger_kernel.transformers import LigerSwiGLUMLP
from liger_kernel.transformers import liger_rotary_pos_emb
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from config import create_model_args

# Create default model args instance
model_args = create_model_args()
# import numpy as np
class RotaryEmbeddings(nn.Module):
    def __init__(
        self,
         device,
        embeddings_dims: int = model_args.embeddings_dims,
        block_size: int = model_args.block_size,
        batch_size: int = model_args.batch_size
    ):
        super().__init__()

        self.embeddings_dims = embeddings_dims
        self.block_size = block_size
        self.batch_size = batch_size
        self.theta = 0
        self.device=device

    
    def apply_rope(self, seq, q=None, k=None):
        batch_size, seq_len, embeds_dims = seq.shape

        if(model_args.use_liger):
          token_idx = torch.arange(0 , seq_len, device = self.device).unsqueeze(1)
          positions = torch.arange(0 , embeds_dims, device = self.device).unsqueeze(0)
          # dims = torch.arange(1, self.embeddings_dims // 2,  dtype=torch.float32)
          theta = 10000 ** (-2 * (positions) / embeds_dims)
          angles = token_idx * theta
          angles = angles.expand(seq_len, -1) # because this thing needs to be applied to every sequence in the batch but with embeds dims halved

          cos = torch.cos(angles)
          sin = torch.sin(angles)
          cos = cos.unsqueeze(0)
          sin = sin.unsqueeze(0)
          # print(cos.shape)
          # print(sin.shape)
          out = liger_rotary_pos_emb(q, k, cos, sin)

        else:

          # print(seq.shape)
          # print(self.embeddings_dims)
          # self.matrix = torch.zeros((seq_len, self.embeddings_dims, self.embeddings_dims), dtype=torch.float32,  requires_grad=False,  device = self.device)
          token_idx = torch.arange(0 , seq_len, device = self.device).unsqueeze(1)
          positions = torch.arange(0 , embeds_dims, 2, device = self.device).unsqueeze(0)
          # dims = torch.arange(1, self.embeddings_dims // 2,  dtype=torch.float32)
          theta = 10000 ** (-2 * (positions) / embeds_dims)
          angles = token_idx * theta
          angles = angles.expand(seq_len, -1) # because this thing needs to be applied to every sequence in the batch but with embeds dims halved
          x_reshaped = seq.view(batch_size, seq_len, embeds_dims // 2, 2)
          
          cos_angles = torch.cos(angles)
          sin_angles = torch.sin(angles)
          # print(cos_angles.shape)
          # print(sin_angles.shape)
          # print(x_reshaped.shape)
          # indices = torch.arange(self.embeddings_dims,  dtype=torch.int64,  device = self.device)

          out = torch.stack([x_reshaped[..., 0]*cos_angles - (x_reshaped[...,1] * sin_angles), x_reshaped[...,1] * cos_angles + x_reshaped[..., 0] * sin_angles], dim=-1)
          out = out.view(batch_size, seq_len, embeds_dims)
        return out

    def forward(self, x, q=None, k=None):
        # print("X shape: ", x.shape)
        # print("X is: ", x)
        # B,T,C = x.shape
        # print("MATRIX:",x)
        # if(x > self.block_size or x < self.block_size):
        #     matrix = self.init_matrix(x)
        #     return matrix
        # else:
        #     matrix = self.init_matrix(self.block_size)

        #     return matrix
        # if(model_args.inference):
        res = self.apply_rope(x, q, k)
        return res 
        # else:
            # return self.x_reshaped
    
class RotaryAttentionHead(nn.Module):
    def __init__(
        self,
         device,
        embeddings_dims: int = model_args.embeddings_dims,
        no_of_heads: int = model_args.no_of_heads,
        attn_dropout: int = model_args.attn_dropout
    ):
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  bias=False,  device = device)
        self.key = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  bias=False,  device = device)
        self.value = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  bias=False, device = device)
        self.rope = RotaryEmbeddings(embeddings_dims=self.head_size,  device = device)
        self.dropout = nn.Dropout(p = attn_dropout)
        self.device = device
    def forward(self,x):
        # print(x.shape)
        # print("X is: ", x)
        batch, block_size, embeddings_dims = x.shape
        query = self.query(x)
        # print(query)
        key = self.key(x)
        values = self.value(x)
        # matrix = self.rotary_matrix(block_size)
        if(model_args.use_liger == False):
          rotary_q = self.rope(query)
          rotary_k = self.rope(key)
        else:

          rotary_q, rotary_k = self.rope(x, query, key)
        # print(matrix.shape)
        # print(query.shape)
        masked = torch.tril(torch.ones((block_size, block_size),  requires_grad=False,  device = self.device))
        # rotary_query = matrix @ query.permute(1,2,0) # (B,T, C,C) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
        # rotary_key = matrix @ key.permute(1,2,0)  #  (B,T, C,C  ) @ (B,T,C) -> (B,C,T) = (B,T,C,T)
        weights = rotary_q.permute(2,0,1) @ rotary_k.permute(2,0,1).transpose(-2, -1)#(B,T,C,T) @ (B,T,C,T) = (T,C,C,T)
        weights_masked = weights.masked_fill(masked == 0, float('-inf'))
        scaled_weights = weights_masked / (torch.sqrt(torch.tensor(key.shape[-1])))
        scaled_weights = F.softmax(scaled_weights, dim=-1)
        value = scaled_weights @ values
        out = self.dropout(value)
        return out

# Text embeddings
class TextEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size = model_args.vocab_size,
        embeddings_dims = model_args.embeddings_dims,
        device = model_args.device
    ):
        super().__init__()
        self.embeddings_table = nn.Embedding(num_embeddings = vocab_size, embedding_dim=embeddings_dims, device=device) #Just a look up table to convert the toekns_ids to some numbers
        # nn.init.normal_(self.embeddings_table.weight.data, mean=0, std=0.02)

    def forward(self, x):
        return self.embeddings_table(x)

#Layer Normalization

class LayerNormalization(nn.Module):
    def __init__(
        self,
        embeddings_dims = model_args.embeddings_dims
    ):
        super().__init__()
        if(model_args.use_liger == False):
            self.norm = nn.LayerNorm(normalized_shape=embeddings_dims)
        else:
            self.norm = LigerLayerNorm(embeddings_dims)

    def forward(self, x):

        return self.norm(x)

class Swish(nn.Module):
    def __init__(
        self,
        block_size: int = model_args.block_size,
        embeddings_dims: int = model_args.embeddings_dims,
        device = model_args.device
    ):
        super().__init__()

        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        swish = x * self.sig(x)

        return swish

class SWiGLUExpertMoE(nn.Module):
    def __init__(
        self,
        block_size: int = model_args.block_size,
        embeddings_dims: int = model_args.embeddings_dims,
        device = model_args.device
    ):
        super().__init__()

        self.hidden_dims = embeddings_dims * 2  #Apply this when memory permits

        if(model_args.use_liger):

          @dataclass
          class config:

              hidden_size = embeddings_dims
              intermediate_size = self.hidden_dims
              hidden_act = 'swish'

          conf = config()

          self.swiglu = LigerSwiGLUMLP(conf)
        else:
          self.swish = Swish(block_size=block_size, embeddings_dims=embeddings_dims, device=device)
          self.linear_layer1 = nn.Linear(in_features=embeddings_dims, out_features=self.hidden_dims,  bias=False, device = device)
          self.linear_layer2 = nn.Linear(in_features=embeddings_dims, out_features=self.hidden_dims,  bias=False, device = device)
          self.linear_layer3 = nn.Linear(in_features=self.hidden_dims, out_features=embeddings_dims,  bias=False, device = device)

    def forward(self, x):
        if(model_args.use_liger == False):
          swish_res = self.swish(self.linear_layer1(x))
          x_V = self.linear_layer2(x)
          res = torch.mul(swish_res, x_V)
          out = self.linear_layer3(res)

        else:
          out = self.swiglu(x)
          # out = self.linear_layer2(out)
          # out = self.linear_layer3(out)
        return out

#MoE Layer

class MoeLayer(nn.Module):
    def __init__(
        self,
        dropout = model_args.dropout,
        embeddings_size = model_args.embeddings_dims,
        device = model_args.device,
        # inner_dimensional_states: int = 3072
    ):
        super().__init__()

        self.heads = nn.ModuleList([SWiGLUExpertMoE() for _ in range(model_args.experts)])
        self.gate = nn.Linear(in_features=embeddings_size, out_features=model_args.experts, device=device, bias=False)
        if(model_args.noisy_topk is True and model_args.use_checkpointing == False):
            self.noise = nn.Linear(in_features=embeddings_size, out_features=model_args.experts, device=device, bias=False)
        # self.outputs = torch.zeros((batch_size,block_size, embeddings_size), device=device) #batch size needs to be defined because we are accessing it explicitly
        self.device = device
    def forward(self, x):
        # mlp_weights_init = self.mlp.apply(weights_init)
        self.gate_out = self.gate(x) #[bz, seq, num_experts]
        if(model_args.noisy_topk == True and model_args.use_checkpointing == False):
            noise = self.noise(x)
            gaussian_noise = torch.normal(0, 1, size=self.gate_out.shape, device=self.device)
            noisy_router = F.softplus(noise) * gaussian_noise
            noisy_router += self.gate_out
        else:
            noisy_router = self.gate_out
        top_k_values, top_k_indices = torch.topk(noisy_router, k=model_args.top_experts) #[bs, seq len, top k]
        probs = torch.nn.functional.softmax(top_k_values, dim=-1) #[bs, seq len, top k]

        out = 0

        out = torch.zeros_like(x)
        for expert_idx in range(model_args.experts):
            # Create mask for current expert across all top_k positions
            expert_mask = (top_k_indices == expert_idx)
            
            # Sum probabilities for current expert
            expert_weights = (probs * expert_mask).sum(dim=-1)  # [batch, seq_len]
            
            # Get inputs where expert is used
            selected = expert_weights > 0
            if not selected.any():
                continue
                
            # Process all selected inputs through expert
            expert_out = self.heads[expert_idx](x[selected])
            
            # Weight and accumulate outputs
            out[selected] += expert_out * expert_weights[selected].unsqueeze(-1)

        return out

class AttentionHead(nn.Module):
    def __init__(
        self,
        attn_dropout = model_args.attn_dropout,
        embeddings_dims = model_args.embeddings_dims,
        no_of_heads = model_args.no_of_heads,
        device = model_args.device
    ):
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        self.no_of_heads = no_of_heads
        if(model_args.use_flash_attention==False):
            self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=model_args.device, bias=False)
            self.keys = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,device=model_args.device, bias=False)
            self.values = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=model_args.device,bias=False)
        # self.dropout = nn.Dropout(p = attn_dropout)
          
        if(model_args.use_flash_attention):
            # Combined linear projections for Q, K, V
            self.qkv_proj = nn.Linear(embeddings_dims, 3 * embeddings_dims, bias=False, device=model_args.device)
        self.dropout = nn.Dropout(p = attn_dropout)
        self.device = device
        if(model_args.use_flash_attention == False):
            self.rotary= RotaryEmbeddings(embeddings_dims=self.head_size,  device = device)
        if(model_args.use_flash_attention):
            self.rotary= RotaryEmbeddings(embeddings_dims=embeddings_dims,  device = device)
        if(model_args.use_liger):
            self.rope = RotaryEmbeddings(embeddings_dims=embeddings_dims,  device = device)
            
    def forward(self, x):
        batch_size, block_size, embd_dims = x.shape
        if(model_args.use_flash_attention == False):
          if(model_args.use_liger == False):
            k = self.keys(x)
            q = self.query(x)
            v = self.values(x)
            q = self.rope(q)
            k = self.rope(k)
            masked_table = torch.tril(torch.ones(block_size, block_size, device=model_args.device))
            weights = q @ torch.transpose(k, dim0=-2, dim1=-1) * (k.shape[-1] ** -0.5)
            masked_values = weights.masked_fill(masked_table[: block_size, : block_size] == 0, float('-inf'))
            weights_normalized = nn.functional.softmax(masked_values, dim=-1) #Normalize along the embeddings dimension for all the tokens
            weights_normalized = self.dropout(weights_normalized)
            out = weights_normalized @ v
            return out
          # else:
            

        else:
            qkv = self.qkv_proj(x)
            q, k, v = qkv.chunk(3, dim=-1)
            # k = self.rotary(k)
            # q = self.rotary(q)
            q = q.view(batch_size, block_size, self.no_of_heads, self.head_size).transpose(1, 2)
            k = k.view(batch_size, block_size, self.no_of_heads, self.head_size).transpose(1, 2)
            v = v.view(batch_size, block_size, self.no_of_heads, self.head_size).transpose(1, 2)
            q, k = self.rope(x, q, k)
            # print(q.shape)
            # print(k.shape)
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=model_args.dropout, is_causal=True
            )

            # Properly merge heads
            out = out.transpose(1, 2).contiguous().view(batch_size, block_size, -1)
            return out

# MHA

class MHA(nn.Module):
    def __init__(
        self,
        attn_dropout = model_args.attn_dropout,
        embeddings_dims = model_args.embeddings_dims,
        no_of_heads = model_args.no_of_heads,
        device = model_args.device
    ):
        super().__init__()
        self.no_of_heads = no_of_heads
        self.heads = nn.ModuleList([AttentionHead(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads, device=device) for _ in range(no_of_heads)])
        self.dropout = nn.Dropout(p = attn_dropout)
        self.linear = nn.Linear(in_features=self.no_of_heads * embeddings_dims, out_features=embeddings_dims, device=device, bias=False) # 12 (no of heads) * (batch_size) 64 = 768 -> gives out the text embeddings

    def forward(self, x):
        concat = torch.cat([head(x) for head in self.heads], dim=-1)
        linear_layer = self.linear(concat)
        out = self.dropout(linear_layer)
        return out

# Decoder Block

class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        attn_dropout = model_args.attn_dropout,
        embeddings_dims = model_args.embeddings_dims,
        no_of_heads = model_args.no_of_heads,
        dropout = model_args.dropout,
        vocab_size = model_args.vocab_size,
        device = model_args.device   
    ):
        super().__init__()

        self.mha = MHA(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads, device=device)
        self.layer_norm1 = LayerNormalization(embeddings_dims=embeddings_dims)
        self.layer_norm2 = LayerNormalization(embeddings_dims=embeddings_dims)
        self.moe_block = MoeLayer(dropout=dropout, embeddings_size=embeddings_dims, device=device)

    def forward(self, x):
        # x = self.mha(x)
        # x = x + self.layer_norm1(x)
        # x = x + self.mlp_block(x)
        # out = self.layer_norm2(x)
        x = x + self.mha(self.layer_norm1(x))  #Very important step -> Layer Norm on input and then passes it to the subsequent blocks
        x = x + self.moe_block(self.layer_norm2(x)) #Very important step

        return x

# Decoder Block

class Mixtral(nn.Module):
    def __init__(
        self,
        attn_dropout = model_args.attn_dropout,
        embeddings_dims = model_args.embeddings_dims,
        no_of_heads = model_args.no_of_heads,
        block_size = model_args.block_size,
        dropout = model_args.dropout,
        no_of_decoder_layers = model_args.no_of_decoder_layers,
        vocab_size = model_args.vocab_size,
        device = model_args.device,
        tokenizer = None
    ):
        super().__init__()

        # Store tokenizer for use in loss calculation
        self.tokenizer = tokenizer
        
        # self.positional_embeddings = nn.Parameter(torch.randn(1, block_size, embeddings_dims, device=device), requires_grad=True) #To give positional embeddings to each token of the input text, hence num_embeddings=block_size
        # torch.nn.init.kaiming_normal_(self.positional_embeddings)
        self.text_embds = TextEmbeddings(vocab_size=vocab_size, embeddings_dims=embeddings_dims, device=device)
        self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=vocab_size, device=device, bias=False) # Takes in logits of dimensions- embeds_dims and converts it into dimension of vocab_size (logits in range of vocab_size)
        self.layer_norm = LayerNormalization(embeddings_dims=embeddings_dims)
        self.decoder_layers = nn.ModuleList([TransformerDecoderBlock(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads, dropout=dropout, vocab_size=vocab_size, device=device) for _ in range(no_of_decoder_layers)])
        self.apply(self.kaiming_init_weights)
        
        # Initialize loss function with tokenizer pad token id if tokenizer is provided
        if self.tokenizer is not None:
            self.le_loss = LigerFusedLinearCrossEntropyLoss(
                ignore_index=self.tokenizer.pad_token_id
            ).to(model_args.device)
       
    def kaiming_init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Embedding):
            torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x, actual_labels = None, inference=False):
        x = self.text_embds(x)
        # x = x + self.positional_embeddings[: , :x.shape[1], :] #@@@Important remember
        for layer in self.decoder_layers:
            if(model_args.use_checkpointing):
                x = checkpoint(layer, x)
            else:
                x = layer(x)
        # x = 2  * ((1.0 / math.sqrt(model_args.no_of_decoder_layers))) * x
        x = self.layer_norm(x)
        if(inference):
            out = self.linear_layer(x)
            return out
        if(model_args.use_liger):  
            # print("yo")
            y = x.contiguous().view(-1, model_args.embeddings_dims)
            if(actual_labels is not None):
                labels = actual_labels.contiguous().view(-1)
                
                # Pass linear layer weights FIRST as required [2][5]
                loss = self.le_loss(self.linear_layer.weight, y, labels)
                return loss
        else:
            # print("Hi")
            out = self.linear_layer(x)
            return out
        
        # out = self.linear_layer(x)
        # return out

def find_unused_parameters(model):
    unused = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused.append(name)
    return unused
