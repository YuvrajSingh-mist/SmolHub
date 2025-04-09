import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ModelArgs:
    device: str = 'cuda'
    epochs: int = 5
    max_lr: float = 2e-5
    rank: int = 4


class LoRALayer(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()


        self.rank = ModelArgs.rank
        self.model_weight_dims = model.config.n_embd
        self.query_A = nn.Parameter(torch.ones((self.model_weight_dims, self.rank), requires_grad=True))
        self.query_B = nn.Parameter(torch.zeros((self.rank, self.model_weight_dims), requires_grad=True))
        self.key_A = nn.Parameter(torch.ones((self.model_weight_dims, self.rank), requires_grad=True))
        self.key_B = nn.Parameter(torch.zeros((self.rank, self.model_weight_dims), requires_grad=True))
        self.value_A = nn.Parameter(torch.ones((self.model_weight_dims, self.rank), requires_grad=True))
        self.value_B = nn.Parameter(torch.zeros((self.rank, self.model_weight_dims), requires_grad=True))
        self.output_A = nn.Parameter(torch.ones((self.model_weight_dims, self.rank), requires_grad=True))
        self.output_B = nn.Parameter(torch.zeros((self.rank, self.model_weight_dims), requires_grad=True))

        torch.nn.init.normal_(self.query_A, mean=0.0, std=1)
        torch.nn.init.normal_(self.key_A, mean=0.0, std=1)
        torch.nn.init.normal_(self.output_A, mean=0.0, std=1)
        torch.nn.init.normal_(self.value_A, mean=0.0, std=1)


    def forward(self, w_o, q_o, k_o, v_o):

        final_weight_WO = w_o + self.output_B.T @ self.output_A.T
        final_weight_QO = q_o + self.query_B.T @ self.query_A.T
        final_weight_KO = k_o + self.key_B.T @ self.key_A.T
        final_weight_VO = v_o + self.value_B.T @ self.value_A.T
      
        return final_weight_WO, final_weight_QO , final_weight_KO, final_weight_VO


class LoRAWrapper(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.lora_layer = LoRALayer(model)
        self.config = model.config
        self.model = model
    def forward(self, x):
        qkv_layers = [self.model.transformer.h[i].attn.c_attn for i in range(self.config.n_layer)]
        o_layers = [self.model.transformer.h[i].attn.c_proj for i in range(self.config.n_layer)]

        for i in range(len(qkv_layers)):
            hidden_size = qkv_layers[i].weight.size(-1) // 3
            Q, K, V = torch.split(qkv_layers[i].weight, hidden_size, dim=-1)
            O = o_layers[i].weight
            out_o, out_q, out_k, out_v = self.lora_layer(O,Q,K,V)
            combined_qkv = torch.concat([out_q, out_k, out_v], dim=-1)
            qkv_layers[i].weight.data.copy_(combined_qkv)
            o_layers[i].weight.data.copy_(out_o)
        return x


class LoRAModel(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.lora_wrapper = LoRAWrapper(model)

        self.config = model.config

    def forward(self,x):

        out = self.lora_wrapper(x)

        return out

