import argparse
from dataclasses import dataclass

def get_args():
    parser = argparse.ArgumentParser(description='SmolKimi - DeepSeek V3 Inspired Model Training')
    
    # Model Architecture
    parser.add_argument('--block_size', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=256, help='Training batch size')
    parser.add_argument('--embeddings_dims', type=int, default=384, help='Model embedding dimensions')
    parser.add_argument('--no_of_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--no_of_decoder_layers', type=int, default=6, help='Number of decoder layers')
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension for attention')
    
    # MoE Configuration
    parser.add_argument('--experts', type=int, default=8, help='Number of MoE experts')
    parser.add_argument('--top_experts', type=int, default=2, help='Number of experts to route to (top-k)')
    parser.add_argument('--use_shared_expert', action='store_true', default=True, help='Enable shared expert in MoE')
    parser.add_argument('--noisy_topk', action='store_true', default=False, help='Use noisy top-k routing')
    parser.add_argument('--useauxFreeLoadBalancingLoss', action='store_true', default=True, help='Use auxiliary-free load balancing loss')
    parser.add_argument('--aux_free_bias_update_rate', type=float, default=0.001, help='Bias update rate for load balancing')
    parser.add_argument('--loss_scale', type=float, default=0.3, help='Loss scaling factor')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--max_lr', type=float, default=6e-4, help='Maximum learning rate')
    parser.add_argument('--weight_decay_optim', type=float, default=0.1, help='Weight decay for optimizer')
    parser.add_argument('--beta_1', type=float, default=0.9, help='Beta1 for optimizer')
    parser.add_argument('--beta_2', type=float, default=0.95, help='Beta2 for optimizer')
    parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon for optimizer')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping value')
    
    # Regularization
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--attn_dropout', type=float, default=0.1, help='Attention dropout rate')
    
    # System Configuration
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--use_checkpointing', action='store_true', default=False, help='Use gradient checkpointing')
    parser.add_argument('--use_liger', action='store_true', default=True, help='Use Liger kernels for optimization')
    parser.add_argument('--ignore_pad_token_in_loss', action='store_true', default=True, help='Ignore padding tokens in loss calculation')
    
    # Data Configuration
    parser.add_argument('--vocab_size', type=int, default=32000 + 1 , help='Vocabulary size (updated based on tokenizer)')
    parser.add_argument('--base_freq', type=int, default=100000, help='Base frequency for positional encoding')
    parser.add_argument('--hf_token', type=str, default=None, help='Hugging Face token for accessing gated models like Llama-2')
    
    # Dataset Selection
    parser.add_argument('--dataset', type=str, default='tinystories', choices=['tinystories', 'fineweb', 'tinyshakespeare'], help='Dataset to use for training')
    
    # Generation Parameters
    parser.add_argument('--generation_max_length', type=int, default=50, help='Maximum length for text generation')
    parser.add_argument('--generation_top_k', type=int, default=50, help='Top-k value for sampling during generation')
    parser.add_argument('--generation_temperature', type=float, default=1.0, help='Temperature for sampling during generation')
    
    # Logging and Checkpointing
    parser.add_argument('--log_interval', type=int, default=100, help='Steps between logging')
    parser.add_argument('--save_interval', type=int, default=2000, help='Steps between saving checkpoints')
    parser.add_argument('--eval_interval', type=int, default=400, help='Steps between evaluation')
    parser.add_argument('--eval_iters', type=int, default=400, help='Number of iterations for evaluation')
    parser.add_argument('--warmup_iters', type=int, default=400, help='Number of warmup iterations')
    parser.add_argument('--total_iters', type=int, default=10000, help='Total training iterations')
    parser.add_argument('--lr_decay_iters', type=int, default=10000, help='Learning rate decay iterations')
    parser.add_argument('--wandb_project', type=str, default='smolkimi', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name')
    
    # Batch Size Configuration
    parser.add_argument('--total_batch_size', type=int, default=524288, help='Total batch size for gradient accumulation')
    parser.add_argument('--micro_batch_size', type=int, default=None, help='Micro batch size (defaults to batch_size)')
    
    # Distributed Training
    parser.add_argument('--use_ddp', action='store_true', default=False, help='Use distributed data parallel')
    
    return parser.parse_args()

@dataclass
class ModelArgs:
    def __init__(self, args=None):
        if args is None:
            args = get_args()
        
        # Model Architecture
        self.block_size = args.block_size
        self.batch_size = args.batch_size
        self.embeddings_dims = args.embeddings_dims
        self.no_of_heads = args.no_of_heads
        self.no_of_decoder_layers = args.no_of_decoder_layers
        self.latent_dim = args.latent_dim
        
        # MoE Configuration
        self.experts = args.experts
        self.top_experts = args.top_experts
        self.use_shared_expert = args.use_shared_expert
        self.noisy_topk = args.noisy_topk
        self.useauxFreeLoadBalancingLoss = args.useauxFreeLoadBalancingLoss
        self.aux_free_bias_update_rate = args.aux_free_bias_update_rate
        self.loss_scale = args.loss_scale
        
        # Training Hyperparameters
        self.epochs = args.epochs
        self.max_lr = args.max_lr
        self.weight_decay_optim = args.weight_decay_optim
        self.beta_1 = args.beta_1
        self.beta_2 = args.beta_2
        self.eps = args.eps
        self.clip = args.clip
        
        # Regularization
        self.dropout = args.dropout
        self.attn_dropout = args.attn_dropout
        
        # System Configuration
        self.device = args.device
        self.use_checkpointing = args.use_checkpointing
        self.use_liger = args.use_liger
        self.ignore_pad_token_in_loss = args.ignore_pad_token_in_loss
        
        # Data Configuration
        self.vocab_size = args.vocab_size
        self.base_freq = args.base_freq
        self.hf_token = args.hf_token
        self.dataset = args.dataset
        
        # Generation Parameters
        self.generation_max_length = args.generation_max_length
        self.generation_top_k = args.generation_top_k
        self.generation_temperature = args.generation_temperature
        
        # Logging and Checkpointing
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.eval_interval = args.eval_interval
        self.eval_iters = args.eval_iters
        self.warmup_iters = args.warmup_iters
        self.total_iters = args.total_iters
        self.lr_decay_iters = args.lr_decay_iters
        self.wandb_project = args.wandb_project
        self.wandb_run_name = args.wandb_run_name
        
        # Batch Size Configuration
        self.total_batch_size = args.total_batch_size
        self.micro_batch_size = args.micro_batch_size if args.micro_batch_size else args.batch_size
        self.gradient_accumulation_steps = self.total_batch_size // (self.micro_batch_size * self.block_size)
        
        # Calculated parameters
        self.min_lr = 0.1 * self.max_lr
        self.save_checkpoint_iter = self.save_interval
        self.eval_check = self.eval_interval
        
        # Distributed Training
        self.use_ddp = args.use_ddp
