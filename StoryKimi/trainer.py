import time
import torch
import torch.nn.functional as F
import math
import tqdm 
import os
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Muon optimizer import
from muon import MuonWithAuxAdam

# Local imports
from config import ModelArgs, get_args
from model import DeepSeekV3
from data import prepare_dataset, initialize_tokenizer
from inference import topk_sampling, save_text
from tokenizer import Tokenizer

def setup_ddp():
    """Initialize DDP setup"""
    # Initialize the process group
    dist.init_process_group(backend='nccl')
    
    # Get local rank from environment variable set by torchrun
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    return local_rank, world_size, rank, device

def cleanup_ddp():
    """Clean up DDP"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_lr(it, model_args):
    # 1) linear warmup for warmup_iters steps
    if it < model_args.warmup_iters:
        return model_args.max_lr * (it + 1) / (model_args.warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > model_args.lr_decay_iters:
        return model_args.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - model_args.warmup_iters) / (model_args.lr_decay_iters - model_args.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return model_args.min_lr + coeff * (model_args.max_lr - model_args.min_lr)


def find_unused_parameters(model):
    unused = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused.append(name)
    return unused


# import tqdm 
def train():
    # Parse command line arguments
    args = get_args()
    model_args = ModelArgs(args)
    
    # Initialize tokenizer with optional HF token
    tokenizer = initialize_tokenizer(model_args.hf_token)
    
    # Initialize DDP if running with torchrun
    use_ddp = 'RANK' in os.environ or model_args.use_ddp
    if use_ddp:
        local_rank, world_size, rank, device = setup_ddp()
        print(f"Rank {rank}/{world_size} on device {device}")
    else:
        device = model_args.device
        rank = 0
        world_size = 1
        print(f"Start running training on {device}.")
    
    # Initialize wandb for experiment tracking (only on rank 0)
    if rank == 0:
        wandb.init(
            project = model_args.wandb_project,
            name = model_args.wandb_run_name,
            config = {
                'ignore_pad_token_in_loss': model_args.ignore_pad_token_in_loss,
                'use_liger': model_args.use_liger,
                'batch_size': model_args.batch_size,
                'embeddings_dims': model_args.embeddings_dims,
                'no_of_decoder_layers': model_args.no_of_decoder_layers,
                'experts': model_args.experts,
                'top_experts': model_args.top_experts,
                'use_shared_expert': model_args.use_shared_expert,
                'world_size': world_size,
                'dataset': model_args.dataset,
                'max_lr': model_args.max_lr,
                'block_size': model_args.block_size
            }
        )
    
    # Create model and move to GPU
    model = DeepSeekV3(embeddings_dims=model_args.embeddings_dims, block_size=model_args.block_size, vocab_size=model_args.vocab_size, dropout=model_args.dropout, device=device)
    model = model.to(device)
    
    # Wrap model with DDP if using distributed training
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Get the underlying model for parameter groups (important for DDP)
    base_model = model.module if use_ddp else model
    
    hidden_weights = [p for p in base_model.decoder.parameters() if p.ndim >= 2]
    hidden_gains_biases = [p for p in base_model.decoder.parameters() if p.ndim < 2]
    nonhidden_params = [*base_model.linear_layer.parameters(), *base_model.embedding.parameters()]
    param_groups = [
        dict(params=hidden_weights, use_muon=True,
             lr=0.02, weight_decay=0.01),
        dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
             lr=model_args.max_lr, betas=(model_args.beta_1, model_args.beta_2), weight_decay=model_args.weight_decay_optim),
    ]
    optimizer = MuonWithAuxAdam(param_groups)

    if rank == 0:
        print("Model loaded")
    # Setup optimizer
    # optimizer = torch.optim.AdamW(params=model.parameters(), lr=ModelArgs.max_lr, betas=(ModelArgs.beta_1, ModelArgs.beta_2), weight_decay=ModelArgs.weight_decay_optim, eps=ModelArgs.eps)
    
    # Training parameters
    # save_checkpoint_iter = 2000
    # total_iters = 610000
    # eval_iters = 1000
    model = torch.compile(model)
    
    # Training progress bar (only on rank 0)
    if rank == 0:
        train_epoch_iterator = tqdm.tqdm(range(ModelArgs.total_iters), desc="Training")
    else:
        train_epoch_iterator = range(ModelArgs.total_iters)
        
    val_dataloader = prepare_dataset('val', device, ModelArgs.batch_size, use_ddp=use_ddp)
    val_iterator = iter(val_dataloader)
    # Get batches for training
    @torch.inference_mode()
    def estimate_loss():
        out = {}
        model.eval()
        count = 0
        for split in ['val']:
            if rank == 0:
                print(f"Starting with {split} evaluation...")
            losses = torch.zeros(ModelArgs.eval_iters, device=device)
            for k in range(ModelArgs.eval_iters):

                nonlocal val_iterator
                
                # for k, batch in enumerate(dataloader):
                try:
                    batch = next(val_iterator)
                except StopIteration:
                    val_iterator = iter(val_dataloader)
                    batch = next(val_iterator)
            
                idx = batch["input_ids"].to(device)
                targets = batch["labels"].to(device)
                # mask = torch.ones(ModelArgs.batch_size, ModelArgs.block_size, dtype=idx.dtype).to(device)  # Create a mask of ones for the entire block
                # mask = mask.masked_fill(idx == tokenizer.pad_token_id, 0)  # Set padding tokens to 0 in the mask
                # if ModelArgs.use_liger:
                #     # Pass actual labels to the model to use optimized loss function
                #     # ignore_index is already set in the model's le_loss initialization
                #     loss = model(idx, actual_labels=targets)
                # else:
                # Standard cross entropy path
                # mask= torch.ones(ModelArgs.batch_size, ModelArgs.block_size, dtype=idx.dtype).to(device)  # Create a mask of ones for the entire block
                # mask = mask.masked_fill(idx == tokenizer.pad_token_id, 0)  # Set padding tokens to 0 in the mask
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Get the base model for DDP
                    eval_model = model.module if use_ddp else model
                    if ModelArgs.use_liger:
                        # Use Liger fused linear cross entropy loss
                        decoder_out = eval_model.decoder(eval_model.embedding(idx) + eval_model.pos_embeddings(eval_model.embedding(idx)), mask=None)
                        decoder_out_flat = decoder_out.contiguous().view(-1, ModelArgs.embeddings_dims)
                        targets_flat = targets.contiguous().view(-1)
                        # Make sure le_loss exists
                        if hasattr(eval_model, 'le_loss'):
                            loss = eval_model.le_loss(eval_model.linear_layer.weight, decoder_out_flat, targets_flat)
                        else:
                            # Fallback to standard cross entropy
                            logits = eval_model.linear_layer(decoder_out)
                            B, T, C = logits.shape
                            logits_flat = logits.contiguous().view(-1, C)
                            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=tokenizer.pad_token_id, reduction='mean')
                    else:
                        # Standard cross entropy loss
                        logits = eval_model(idx, mask=None)  # Get logits from the model [B, T, C]
                        B, T, C = logits.shape
                        
                        # Standard next-token prediction loss
                        logits_flat = logits.contiguous().view(-1, C)  # [B*T, C]
                        targets_flat = targets.contiguous().view(-1)   # [B*T]
                        
                        # Compute cross entropy loss
                        loss = F.cross_entropy(
                            logits_flat, 
                            targets_flat, 
                            ignore_index=tokenizer.pad_token_id,
                            reduction='mean'
                        )
                
                losses[k] = loss.item()
                # count += 1
            
            # Average losses across all processes if using DDP
            if use_ddp:
                dist.all_reduce(losses, op=dist.ReduceOp.AVG)
                
            out[split] = losses.mean()

        model.train()
        return out
    token_count = 0
    # Start training loop
    model.train()
    if rank == 0:
        print("Lessgoo...")
        print("gradient steps: ", ModelArgs.gradient_accumulation_steps)
    dataloader = prepare_dataset('train', device, ModelArgs.batch_size, use_ddp=use_ddp)
    train_dataloader = iter(dataloader) 
    accumulated_loss = 0.0
    
    # if ModelArgs.use_compile:
    # model = torch.compile(model)
    if rank == 0:
        print("Model compiled")
    
    for epoch in range(ModelArgs.epochs):
        # Set epoch for DistributedSampler
        if use_ddp and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
            
        for step in train_epoch_iterator:
            # Periodically evaluate loss on train and val sets (only on rank 0)
            if ((step % ModelArgs.eval_iters == 0 and step != 0) or step == ModelArgs.total_iters - 1) and rank == 0:
                losses = estimate_loss()
                avg_val_loss = torch.Tensor([losses['val']]).to(device)
                print(f"step {step}: train loss {accumulated_loss:.4f}, val loss {losses['val']:.4f}")
                val_perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
                # Log metrics to wandb
                wandb.log({
                    "val_perplexity": val_perplexity,
                    # "val_step_loss": losses['train'],
                    "val_step_loss": losses['val'],
                    "step": step
                })
            
            # Save checkpoint periodically (only on rank 0)
            if step % ModelArgs.save_checkpoint_iter == 0 and step != 0 and rank == 0:
                print(f"Saving the model checkpoint for step: {step}")
                # Save the base model state dict (unwrapped from DDP)
                save_model = model.module if use_ddp else model
                torch.save(save_model.state_dict(), f"checkpoint_{step}.pt")
                print("Checkpoint saved")
            
            # Initialize gradient accumulation
            accumulated_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            
            # Gradient accumulation loop
            for micro_step in range(ModelArgs.gradient_accumulation_steps):
                # Get batch for training step
                try:
                    batch = next(train_dataloader)
                except StopIteration:
                    train_dataloader = iter(dataloader)
                    batch = next(train_dataloader)
                
                idx = batch['input_ids'].to(device)
                targets = batch['labels'].to(device)
                
                token_count += idx.numel()
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Get the base model for DDP
                    train_model = model.module if use_ddp else model
                    if ModelArgs.use_liger:
                        # Use Liger fused linear cross entropy loss
                        decoder_out = train_model.decoder(train_model.embedding(idx) + train_model.pos_embeddings(train_model.embedding(idx)), mask=None)
                        decoder_out_flat = decoder_out.contiguous().view(-1, ModelArgs.embeddings_dims)
                        targets_flat = targets.contiguous().view(-1)
                        # Make sure le_loss exists
                        if hasattr(train_model, 'le_loss'):
                            loss = train_model.le_loss(train_model.linear_layer.weight, decoder_out_flat, targets_flat)
                        else:
                            # Fallback to standard cross entropy
                            logits = train_model.linear_layer(decoder_out)
                            B, T, C = logits.shape
                            logits_flat = logits.contiguous().view(-1, C)
                            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=tokenizer.pad_token_id, reduction='mean')
                    else:
                        # Standard cross entropy loss
                        logits = train_model(idx, mask=None)  # Get logits from the model [B, T, C]
                        B, T, C = logits.shape
                        
                        # Standard next-token prediction loss
                        logits_flat = logits.contiguous().view(-1, C)  # [B*T, C]
                        targets_flat = targets.contiguous().view(-1)   # [B*T]
                        
                        # Compute cross entropy loss
                        loss = F.cross_entropy(
                            logits_flat, 
                            targets_flat, 
                            ignore_index=tokenizer.pad_token_id,
                            reduction='mean'
                        )
                
                # Scale loss by gradient accumulation steps
                loss = loss / ModelArgs.gradient_accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()
                
                if micro_step % 10 == 0 and rank == 0:
                    print(f"Micro Batch: {micro_step}/{ModelArgs.gradient_accumulation_steps}")
                    print(f"Step: {step}/{ModelArgs.total_iters}")
                    print(f"Total tokens processed: {token_count}")
            
            # Synchronize accumulated loss across all processes if using DDP
            if use_ddp:
                loss_tensor = torch.tensor(accumulated_loss, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                accumulated_loss = loss_tensor.item()
            
            # Update learning rate
            lr = get_lr(step)
            for params in optimizer.param_groups:
                params['lr'] = lr
            
            # Compute gradient norms before clipping
            grad_norm_value = 0.0
            if ModelArgs.clip != 0.0:
                total_norm_before = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2
                )
                grad_norm_value = total_norm_before.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=ModelArgs.clip)
                
                total_norm_after = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2
                )
                
                if step != 0 and rank == 0:
                    print(f"Gradient Norm Before Clipping: {total_norm_before.item():.4f}")
                    print(f"Gradient Norm After Clipping: {total_norm_after.item():.4f}")

            # Optimizer step after accumulating gradients
            optimizer.step()
            
            # Calculate perplexity from accumulated loss
            perplexity = torch.exp(torch.tensor(accumulated_loss)).item()
            
            # Log metrics to wandb (only on rank 0)
            if rank == 0:
                wandb.log({
                    "Learning Rate": optimizer.param_groups[0]['lr'],
                    "Train_Loss": accumulated_loss,
                    "Train Perplexity": perplexity,
                    "Total Tokens Processed": token_count,
                    "Step": step,
                    "Gradient Norm": grad_norm_value,
                    "Gradient Accumulation Steps": ModelArgs.gradient_accumulation_steps
                })
                
                if step % ModelArgs.eval_iters == 0:
                    prompt = "Once upon a time there lived a baby deer named Bambi. "
                    generated_text = topk_sampling(train_model, prompt, max_length=ModelArgs.block_size, top_k=(50 * 2), temperature=0.9, device=device)
                    print(f" Step: {step} | Generated Text: {generated_text}")
                    save_text(f"generated_data/generated_text_{step}.txt", step, generated_text)
                    
        # Clean up DDP
        if use_ddp:
            cleanup_ddp()
            
        # Finish wandb run (only on rank 0)
        if rank == 0:
            wandb.finish()

if __name__ == "__main__":
    train()
