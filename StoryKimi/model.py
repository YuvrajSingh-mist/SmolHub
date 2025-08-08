import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import RMSNorm
from config import ModelArgs
from tokenizer import Tokenizer

# Initialize tokenizer globally as None - will be set later
tokenizer = None
model_args = ModelArgs()


def initialize_tokenizer(hf_token=None):
    """Initialize the global tokenizer with the provided HF token"""
    global tokenizer
    if tokenizer is None:
        tokenizer_instance = Tokenizer(hf_token=hf_token)
        tokenizer = tokenizer_instance.ready_tokenizer()
    return tokenizer

class Normalization(nn.Module):
    def __init__(
        self,
        embeddings_dims: int = model_args.embeddings_dims
    ):  
        super().__init__()
        self.rmsnorm_layer = RMSNorm(embeddings_dims)
        
        
    def forward(self, x):
        
        x = self.rmsnorm_layer(x)
        return x
        


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

        self.hidden_dims = (embeddings_dims * 2)
        self.swish = Swish(block_size=block_size, embeddings_dims=embeddings_dims, device=device)
        self.linear_layer1 = nn.Linear(in_features=embeddings_dims, out_features=self.hidden_dims,  bias=False, device = device)
        self.linear_layer2 = nn.Linear(in_features=embeddings_dims, out_features=self.hidden_dims,  bias=False, device = device)
        self.linear_layer3 = nn.Linear(in_features=self.hidden_dims, out_features=embeddings_dims,  bias=False, device = device)




    def forward(self, x):
        swish_res = self.swish(self.linear_layer1(x))
        x_V = self.linear_layer2(x)
        res = torch.mul(swish_res, x_V)
        out = self.linear_layer3(res)
        return out



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
        
        # Only create shared expert if enabled
        if model_args.use_shared_expert:
            self.shared_expert = SWiGLUExpertMoE()
        else:
            self.shared_expert = None
            
        if(model_args.noisy_topk is True and model_args.use_checkpointing == False):
            self.noise = nn.Linear(in_features=embeddings_size, out_features=model_args.experts, device=device, bias=False)
            self.noisy_router = None
        # self.outputs = torch.zeros((batch_size,block_size, embeddings_size), device=device) #batch size needs to be defined because we are accessing it explicitly
        self.device = device
        # self.shared_expert_out = torch.zeros((model_args.batch_size, model_args.embeddings_dims), device=device)
        # self.b = torch.zeros((model_args.batch_size, model_args.block_size, model_args.experts), device=device)

        if model_args.useauxFreeLoadBalancingLoss:
            self.register_buffer('routing_bias', torch.zeros(model_args.experts, device=self.device))
            # self.routing_bias = torch.zeros(model_args.experts, device=self.device)
            self.bias_update_speed = model_args.aux_free_bias_update_rate
        
        
    def forward(self, x):
        # mlp_weights_init = self.mlp.apply(weights_init)
        self.gate_out = self.gate(x) #[bz, seq, num_experts]

        
        if(model_args.noisy_topk == True and model_args.use_checkpointing == False):
            noise = self.noise(x)
            gaussian_noise = torch.normal(0, 1, size=self.gate_out.shape, device=self.device)
            self.noisy_router = F.softplus(noise) * gaussian_noise
            self.gate_out += self.noisy_router
            
        

        shared_output = 0
        out = 0

        

        if model_args.useauxFreeLoadBalancingLoss:
            
           self.gate_out += self.routing_bias
           
        
                
        
        # Adjust top_k based on whether shared expert is used
        top_k = model_args.top_experts
        top_k_values, top_k_indices = torch.topk(self.gate_out, k=top_k) #[bs, seq len, top k]
        # topkmask = torch.ones_like(top_k_values, device=self.device)  # [bs, seq len, experts]
        # indices = torch.arange(top_k_values.size(0), device=self.device).unsqueeze(1).unsqueeze(2)  # [bs, 1, 1]
        # topkvaluesMasked = top_k_values.masked_fill(indices != top_k_indices, float('-inf'))  # Mask out negative values
        masked = torch.full_like(self.gate_out, float('-1e20'), device=self.device) 
        masked_values = masked.scatter_(-1, top_k_indices, top_k_values)
        probs = torch.nn.functional.softmax(masked_values, dim=-1) #[bs, seq len, top k]
        
        out = torch.zeros_like(x)
        if model_args.use_shared_expert and self.shared_expert is not None:
            shared_output += self.shared_expert(x)

        flat_x = x.view(-1, x.size(-1))  # Flatten the input for easier processing

        for i in range(model_args.experts): # Iterate through each expert index (0 to num_experts-1)
            # Determine which tokens routed to this expert 'i'
            # top_k_indices is [bs, seq_len, self.top_k]
            # We want a mask of shape [bs, seq_len] where True if expert 'i' is in the top_k for that token
            expert_i_is_chosen_mask = (top_k_indices == i).any(dim=-1) # Check along the top_k dimension
            # expert_i_is_chosen_mask has shape [bs, seq_len]

            if not expert_i_is_chosen_mask.any(): # If expert 'i' was not chosen by any token
                continue

            # Flatten the mask to apply to flat_x
            flat_expert_i_is_chosen_mask = expert_i_is_chosen_mask.reshape(-1) # Shape: [bs * seq_len]

            # Select input tokens for this expert
            selected_input_tokens = flat_x[flat_expert_i_is_chosen_mask] # Shape: [num_active_for_expert_i, embed_dim]

            if selected_input_tokens.numel() == 0: # Should be caught by .any() above, but good check
                continue

            # Process through the expert
            expert_output_for_selected = self.heads[i](selected_input_tokens)

            # Get the routing probabilities for these chosen tokens specifically for expert 'i'
            # routing_probs is [bs, seq_len, num_experts]
            # expert_i_probs_original_shape = routing_probs[:, :, i] # Probabilities for expert 'i', shape [bs, seq_len]
            # flat_expert_i_probs = expert_i_probs_original_shape.reshape(-1) # Shape [bs * seq_len]
            # active_token_weights = flat_expert_i_probs[flat_expert_i_is_chosen_mask] # Shape: [num_active_for_expert_i]

            # Alternative way to get weights directly using the mask on routing_probs for expert i:
            # Get the [bs, seq_len] slice of probabilities for the current expert 'i'
            probs_for_expert_i = probs[:, :, i] # Shape: [bs, seq_len]
            # Now use the expert_i_is_chosen_mask (which is also [bs, seq_len]) to select the relevant weights
            active_token_weights = probs_for_expert_i[expert_i_is_chosen_mask] # Shape: [num_active_for_expert_i]


            weighted_expert_output = expert_output_for_selected * active_token_weights.unsqueeze(-1)

            # Add this expert's contribution
            temp_contribution_for_expert_i = torch.zeros_like(x) # Initialize with zeros
            temp_contribution_for_expert_i.masked_scatter_(
                expert_i_is_chosen_mask.unsqueeze(-1).expand_as(x), # Use the original 2D mask, expanded
                weighted_expert_output
            )
            out = out + temp_contribution_for_expert_i
            
            
        # for expert_idx in range(model_args.experts):
        #     # Create mask for current expert across all top_k positions
        #     expert_mask = (top_k_indices == expert_idx)

        #     # Sum probabilities for current expert
        #     expert_weights = (probs * expert_mask).sum(dim=-1)  # [batch, seq_len]

        #     # Get inputs where expert is used
        #     selected = expert_weights > 0
        #     if not selected.any():
        #         continue
        #     # print(expert_weights.shape)
        #     # print(x[selected].shape)

        #     # Process all selected inputs through expert
        #     expert_out = self.heads[expert_idx](x[selected])
            
            
                
        #     # Weight and accumulate outputs
        #     out[selected] += expert_out * expert_weights[selected].unsqueeze(-1)

        out = out + shared_output  # Add shared expert output if enabled
        
        if model_args.useauxFreeLoadBalancingLoss and self.training:
            
            with torch.no_grad():  
                ci = probs.sum(dim=(0,1))  # Su  of tokens for each expert
                ci_avg = ci.mean()
                
                
                error_i = ci_avg - ci
                
                self.update = self.bias_update_speed * torch.sign(error_i)  # Update routing bias
                self.routing_bias.add_(self.update)
                # self.routing_bias = self.routing_bias + self.update

        return out
    
    
# import numpy as np
class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(
        self,
        device,
        embeddings_dims: int = model_args.embeddings_dims,
        block_size: int = model_args.block_size,
        batch_size: int = model_args.batch_size,
    ):
        super().__init__()

        self.embeddings_dims = embeddings_dims
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        
        # Create positional encoding matrix
        pe = torch.zeros(block_size, embeddings_dims)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embeddings_dims, 2).float() * (-math.log(10000.0) / embeddings_dims))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer so it's not a parameter but moves with the model
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: [1, block_size, embeddings_dims]

    def forward(self, x):
        # x shape: [batch_size, seq_len, embeddings_dims]
        batch_size, seq_len, _ = x.shape
        
        # Add positional embeddings
        # pe[:, :seq_len] ensures we only use the positional embeddings up to the sequence length
        pos_emb = self.pe[:, :seq_len].to(x.device)
        return pos_emb



class LatentAttention(nn.Module):
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
        # if(model_args.use_flash_attention==False):
        self.latent_dim = model_args.latent_dim
        self.W_k = nn.Linear(in_features=self.latent_dim, out_features=self.head_size, device=device, bias=False)
        self.W_v = nn.Linear(in_features=self.latent_dim, out_features=self.head_size, device=device, bias=False)
        self.W_dkv = nn.Linear(in_features=model_args.embeddings_dims, out_features=self.latent_dim, device=device, bias=False) # 3 for query, key and value
        self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=model_args.device, bias=False)
        # self.keys = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,device=model_args.device, bias=False)
        # self.values = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=model_args.device,bias=False)
    # self.dropout = nn.Dropout(p = attn_dropout)
        

        self.dropout = nn.Dropout(p = attn_dropout)
        self.device = device

        # Use sinusoidal positional embeddings instead of rotary
        self.pos_embeddings = SinusoidalPositionalEmbeddings(embeddings_dims=self.head_size, device=device)
        # self.register_buffer('absorbed_q', None)
        # self.absorbed_q = None
        
    def forward(self, x, kv_cache=None, mask=None):
        batch_size, block_size, embd_dims = x.shape

        # k = self.keys(x)
        # q = self.query(x)
        # v = self.values(x)
        
        self.latent_matrix = self.W_dkv(x)

        # print("q shape: ", q.shape)
        
        # print("Shape of latent mat: ", self.query.weight.shape)
        # print("Shape of compressed_k: ", self.W_k.weight.shape)
        
        # if(self.absorbed_q is None):
        self.absorbed_q = torch.matmul(self.query.weight.T , self.W_k.weight)
        
        
        # weights = q @ torch.transpose(k, dim0=-2, dim1=-1) * (k.shape[-1] ** -0.5)

        # if kv_cache is None:
        if kv_cache is None:
            kv_cache = self.latent_matrix
        else:
            # print(kv_cache)
            # print("Shape of latent matrix: ", self.latent_matrix.shape)
            # print("Shape of kv_cache: ", kv_cache.shape)
            kv_cache = torch.cat([kv_cache, self.latent_matrix], dim=1)

        self.compressed_k = self.W_k(kv_cache)
        self.compressed_v = self.W_v(kv_cache)
        
        q_res = torch.matmul(x , self.absorbed_q)
        weights =  q_res @ torch.transpose(kv_cache, dim0=-2, dim1=-1) * (self.head_size ** -0.5)  # [batch_size, block_size, block_size]
        # print("Shape of weights: ", weights.shape)
        # print("Shape of kv_cache: ", kv_cache.shape)
        if(mask is not None):
            weights = weights.masked_fill(mask == 0, float('-1e20')) #Masking the attention weights
            
        masked_table = torch.tril(torch.ones(q_res.shape[1], kv_cache.shape[1], device=model_args.device))

        masked_values = weights.masked_fill(masked_table[: q_res.shape[1], : kv_cache.shape[1]] == 0, float('-1e20'))
        weights_normalized = nn.functional.softmax(masked_values, dim=-1) #Normalize along the embeddings dimension for all the tokens
        weights_normalized = self.dropout(weights_normalized)
        
        # print("Shape of weights_normalized: ", weights_normalized.shape)
        # Apply positional embeddings to the output
        
        
        
        
        # print("Shape of compressed_v: ", self.compressed_v.shape)
        out = weights_normalized @ self.compressed_v
        
        # out = self.pos_embeddings(out)
        return out, kv_cache

# MHA


class MHLA(nn.Module):
    def __init__(
        self,
        device,
        attn_dropout = model_args.attn_dropout,
        embeddings_dims = model_args.embeddings_dims,
        no_of_heads = model_args.no_of_heads,
    ):
        super().__init__()
        self.heads = nn.ModuleList([LatentAttention(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads) for _ in range(no_of_heads)])
        self.dropout = nn.Dropout(p = attn_dropout)
        self.linear = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims, device=device, bias=False) # 12 (no of heads) * (batch_size) 64 = 768 -> gives out the text embeddings
        
    def forward(self, x, kv_cache=None, mask=None):
        # concat = torch.cat([head(x, kv_cache=kv_cache, mask=mask) for head in self.heads], dim=-1)
        res = []
        for head in self.heads:
            head_out, kv_cache = head(x, kv_cache=kv_cache, mask=mask)
            res.append(head_out)
        concat = torch.cat(res, dim=-1)  # Concatenate along the last dimension
        linear_layer = self.linear(concat)
        out = self.dropout(linear_layer)
        return out, kv_cache

class FFN(nn.Module):
    def __init__(self,
                  device,
                  embeddings_dims: int = model_args.embeddings_dims,
                  block_size: int = model_args.block_size,
                  vocab_size: int = model_args.vocab_size,
                   dropout = model_args.dropout

                 ):
        super().__init__()

        self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  dtype=torch.float32,  device = device)
        self.linear_layer2 = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  dtype=torch.float32, device = device)

        self.dropout = nn.Dropout(p = dropout)  # Uncommenting the dropout line
    def forward(self, x):

        x = self.linear_layer(x)
        x = F.gelu(x)
        x = self.linear_layer2(x)
        x = F.gelu(x)
        # x = self.dropout(x)  # Uncommenting the dropout line
        return x







class DecoderLayer(nn.Module):
    def __init__(self,
                device,
                attn_dropout: float = model_args.attn_dropout,
                no_of_heads: int = model_args.no_of_heads,
                embeddings_dims: int = model_args.embeddings_dims,
                dropout = model_args.dropout,
                block_size: int = model_args.block_size,
                vocab_size: int = model_args.vocab_size,

                 ) :
        super().__init__()

        # self.base_freq = model_args.base_freq
        # self.feedforward_network = FFN(embeddings_dims=embeddings_dims, block_size=block_size, vocab_size=vocab_size,  device = device)
        self.mha = MHLA(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads, device=device)
        self.layer_norm1 = Normalization(embeddings_dims=embeddings_dims)
        self.layer_norm2 = Normalization(embeddings_dims=embeddings_dims)
        # self.layer_norm3 = Normalization(embeddings_dims=embeddings_dims)
        self.dropout = nn.Dropout(p = dropout)

        self.moe_block = MoeLayer(dropout=dropout, embeddings_size=embeddings_dims)

    def forward(self, x, kv_cache=None, ffn=None, mask=None):

        out, kv_cache = self.mha(self.layer_norm1(x), kv_cache=kv_cache, mask=mask)  #Very important step -> Layer Norm on input and then passes it to the subsequent blocks
        x = x + out  # Fixed: removed in-place operation
        x = x + self.moe_block(self.layer_norm2(x)) #Very important step

        return x, kv_cache


class Block(nn.Module):
    def __init__(self,
                  device,
                  embeddings_dims: int = model_args.embeddings_dims,
                  no_of_decoder_layers: int = model_args.no_of_decoder_layers,
                  block_size: int = model_args.block_size,
                  vocab_size: int = model_args.vocab_size,
                  dropout = model_args.dropout

                 ) :
        super().__init__()
        self.base_freq = model_args.base_freq
        # self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeddings_dims,  dtype=torch.float32,  device = device)
        self.decoder = nn.ModuleList(DecoderLayer(embeddings_dims=embeddings_dims, block_size=block_size, vocab_size=vocab_size, dropout=dropout,  device = device) for _ in range(no_of_decoder_layers))
        # self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=vocab_size,  dtype=torch.float32,  device = device)
        self.dropout = nn.Dropout(p = dropout)
        self.norm = Normalization(embeddings_dims)
        
        #weight tying
        # self.embeddings.weight = self.linear_layer.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)



    def forward(self, x, mask=None, actual_labels = None, inference=False):
        index = 0
        no_of_layers = 0
        # x = self.embeddings(x)
        # # x = self.dropout(x)
        # if(mask is not None):
        kv_cache = None
        #     x = x * mask
        #     # mask = mask.unsqueeze(-1)
        # x = self.decoder(x)
        for layer in self.decoder:
            # if no_of_layers % 2 == 0:
            #     if no_of_layers % 4 == 0:
            #         # print("x shape: ", x.shape)
            #         x = layer(x, rope=False, ffn=True, mask=mask)
            #     x = layer(x, rope=True, ffn=True, mask=mask)
                
            #     # print("x shape: ", x.shape)
            # else:
            #     # print("x shape local: ", x.shape)
            #     if no_of_layers % 4 == 0:
            #         # print("x shape: ", x.shape)
            #         x = layer(x, rope=False, ffn=False, mask=mask)
            x, kv_cache = layer(x, kv_cache=kv_cache, ffn=None, mask=mask)
                # print("x shape local: ", x.shape)
            # no_of_layers += 1
        # print(x.shape)
        x = self.dropout(x)
        x = 2 * ((model_args.no_of_decoder_layers) ** -0.5) * x
        x = self.norm(x)
        
        # if(inference):
        #     out = self.linear_layer(x)
        #     return out
        # if(model_args.use_liger):  
        #     # print("yo")
        #     y = x.contiguous().view(-1, model_args.embeddings_dims)
        #     if(actual_labels is not None):
        #         labels = actual_labels.contiguous().view(-1)
                
        #         # Pass linear layer weights FIRST as required [2][5]
        #         # ignore_index is already set during initialization
        #         loss = self.le_loss(self.linear_layer.weight, y, labels)
        #         return loss
        # else:
        #     # print("Hi")
        #     out = self.linear_layer(x)
        #     return out

        return x



class DeepSeekV3(nn.Module):
    def __init__(self,
                 device,
                 embeddings_dims: int = model_args.embeddings_dims,
                 block_size: int = model_args.block_size,
                 vocab_size: int = model_args.vocab_size,
                 dropout = model_args.dropout
                ):
        super().__init__()
        self.decoder = Block(device=device, embeddings_dims=embeddings_dims, no_of_decoder_layers=model_args.no_of_decoder_layers, block_size=block_size, vocab_size=vocab_size, dropout=dropout)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeddings_dims, dtype=torch.float32, device=device)
        self.pos_embeddings = SinusoidalPositionalEmbeddings(embeddings_dims=embeddings_dims, device=device)
        self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=vocab_size, dtype=torch.float32, device=device, bias=False)
        # Weight tying - tie embedding and output projection weights
        self.embedding.weight = self.linear_layer.weight
        
        # Initialize the LigerFusedLinearCrossEntropyLoss for optimized training
        if model_args.use_liger:
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
            # Initialize with ignore_index for padding tokens if enabled
            if model_args.ignore_pad_token_in_loss:
                self.le_loss = LigerFusedLinearCrossEntropyLoss(
                    ignore_index=tokenizer.pad_token_id
                )
            else:
                self.le_loss = LigerFusedLinearCrossEntropyLoss()

    def forward(self, x, inference=False, mask=None):
        if(mask is not None):
            x = x * mask
            
        x = self.embedding(x)
        x = x + self.pos_embeddings(x)  # Add positional embeddings
        B, T, C = x.shape
            
        if inference:
            # For inference, we only need the last token prediction
            decoder_out = self.decoder(x, mask=mask)
            logits = self.linear_layer(decoder_out)
            return logits
        else:
            decoder_out = self.decoder(x, mask=mask)
            logits = self.linear_layer(decoder_out)
            return logits
