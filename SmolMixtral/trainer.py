import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import wandb
from tqdm import tqdm
import os

from SmolTransformer.tokenizer import Tokenizer
from config import create_model_args, create_tokenizer
from model import Mixtral, find_unused_parameters
from data import prepare_dataset
from inference import topk_sampling, save_to_file



def setup(rank=None, world_size=None):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl")
    # torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    
def cleanup():
    destroy_process_group()

def _save_snapshot(model, optimizer, scheduler, epoch, step):
    snapshot = {
        "MODEL_STATE": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        "OPTIMIZER_STATE": optimizer.state_dict(),
        # "SCHEDULER_STATE": scheduler.state_dict(),  
        "EPOCHS_RUN": epoch,
        "STEP_RUN": step
    }
    torch.save(snapshot, f"checkpoints/snapshot_{step}.pt")
    print(f"Epoch: {epoch} | Step: {step} | Snapshot saved.")

def _load_snapshot(snapshot_path, model, optimizer, scheduler):
    snapshot = torch.load(snapshot_path)
    model.load_state_dict(snapshot["MODEL_STATE"])
    optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
    scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])
    epoch = snapshot["EPOCHS_RUN"]
    step = snapshot["STEP_RUN"]
    
    print(f"Resuming from Epoch {epoch}, Step {step}")
    return epoch, step

# learning rate decay scheduler (cosine with warmup) from https://github.com/karpathy/nanoGPT/blob/master/train.py

class CustomLRScheduler:
    def __init__(self, optimizer, warmup_iters, lr_decay_iters, min_lr, max_lr):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.it = 0
        self._last_lr = [max_lr]  # Initialize with max_lr (matching PyTorch convention)
        
    def step(self):
        
        self._last_lr = [self._get_lr()]  # Store as list
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._last_lr[0]
        self.it += 1

    def get_last_lr(self):
        return self._last_lr  # Returns list to match PyTorch convention
    
    def _get_lr(self):

      
        # 1) linear warmup for warmup_iters steps
        if self.it < self.warmup_iters:
            return self.max_lr * (self.it + 1) / (self.warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if self.it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (self.it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
        return self.min_lr + coeff * (self.max_lr - self.min_lr)
    
    def state_dict(self):
        return {
            'warmup_iters': self.warmup_iters,
            'lr_decay_iters': self.lr_decay_iters,
            'min_lr': self.min_lr,
            'max_lr': self.max_lr,
            'it': self.it
        }
    
    def load_state_dict(self, state_dict):
        self.warmup_iters = state_dict['warmup_iters']
        self.lr_decay_iters = state_dict['lr_decay_iters']
        self.min_lr = state_dict['min_lr']
        self.max_lr = state_dict['max_lr']
        self.it = state_dict['it']

def train(model_args=None, hf_token=None):
    """
    Main training function with proper dependency injection
    """
    # Initialize model args and tokenizer
    if model_args is None:
        model_args = create_model_args(hf_token=hf_token)
    
    # Use the tokenizer from model_args instead of creating a new one
    tokenizer = model_args.tokenizer
    
    # print("dtype: ", model_args.dtype)
    # device='cuda:0'
    
#         # Initialise run
    wandb.init(
            # entity = 'rajceo2031',
                        project = model_args.wandb_project,
                        # config = CFG,
                        # save_code = True,
                        #group = 'ANN',
                        #job_type = 'train'
)
    print("wandb initialized")
    
    model = Mixtral(
        attn_dropout=model_args.attn_dropout, 
        embeddings_dims=model_args.embeddings_dims, 
        no_of_heads=model_args.no_of_heads, 
        block_size=model_args.block_size, 
        dropout=model_args.dropout, 
        no_of_decoder_layers=model_args.no_of_decoder_layers, 
        vocab_size=model_args.vocab_size, 
        device=model_args.device,
        tokenizer=tokenizer
    )
    
    # print(f"Model on device {device} is ready")
    # print(f"Model on device {device} is ready")

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=model_args.max_lr, 
        betas=(model_args.beta_1, model_args.beta_2),
        weight_decay=model_args.weight_decay_optim,
        eps=model_args.eps,
        
    )
    if(model_args.use_compile):
        model = torch.compile(model)

    model = model.to(model_args.device)
    
    # model = DDP(model, device_ids=[device])
    

    
    
    model.eval()
    world_size = torch.cuda.device_count()
    @torch.inference_mode()
    def estimate_loss(val_loader, val_iterator, device):
        out = {}
        # train_loader = prepare_dataset('train', model_args.batch_size)
        
        # val_loader_iterator = iter(val_loader)
        loader = None
        epoch_loss = None
        epoch_losses = []
        # print("Starting the eval...")
        for split in ['val']:
            print(f"Starting with {split} evaluation...")
            # losses = torch.zeros(ModelArgs.val_epochs)
            # if(split == 'train'):
            #         loader = train_loader
            # if(split == 'val'):
            #         loader = val_loader
            for step in range(model_args.eval_check):  
                try:
                    batch = next(val_iterator)
                except StopIteration:
                    val_loader_iterator = iter(val_loader)
                    batch = next(val_loader_iterator)
                
                total_loss = 0  
                # loader.sampler.set_epoch(step)
                total_batches = 0 
                # batch = next(val_loader_iterator)
                # for batch in loader:  # Loop through DataLoader batches
                idx = batch['input_ids']
                targets = batch['labels']
                idx = idx.to(device)
                targets = targets.to(device)
                # with torch.autocast(device_type=model_args.device, dtype=torch.bfloat16):
                if(model_args.use_liger):
                    loss = model(idx, actual_labels = targets)
                else:
                    logits = model(idx)
                    batch_size, block_size, embeddings_dims = logits.shape
                    # print(logits.shape)   
                    # print(targets)
                    logits = logits.view(batch_size*block_size, embeddings_dims)
                    # print("OK")
                    targets = targets.view(batch_size * block_size)
                    # print("OK2")
                    loss = nn.functional.cross_entropy(logits, targets, ignore_index=tokenizer.pad_token_id)
                    # batch_size, block_size, embeddings_dims = logits.shape
                    # logits = logits.view(batch_size * block_size, embeddings_dims)
                    # targets = targets.view(batch_size * block_size)

                    # loss = F.cross_entropy(logits, targets, ignore_index=tokenizer.pad_token_id)

                total_loss += loss.item()
                total_batches += 1

            # Compute mean loss for this epoch
            epoch_loss = total_loss / total_batches if total_batches > 0 else 0.0
            epoch_losses.append(epoch_loss)

                # print(f"Epoch {epoch + 1}/{ModelArgs.val_epochs}: Loss = {epoch_loss:.4f}")

            # Compute mean loss across all evaluation epochs
            out[split] = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            epoch_loss = None
            epoch_losses = []

        model.train()
        return out

    # model = model.to(rank)
    model.train()
    count = 0
   
    train_dataloader = prepare_dataset('train', model_args.device, model_args.batch_size, tokenizer, model_args.block_size, model_args.hf_token)
    val_loader= prepare_dataset('val', model_args.device, model_args.batch_size, tokenizer, model_args.block_size, model_args.hf_token)
    # for step in tqdm(range(total_iters)):
    # for epoch in range(ModelArgs.epochs):
        # torch.cuda.synchronize() 
    
    # train_dataloader.sampler.set_epoch(epoch)
    
    # val_loader.sampler.set_epoch(epoch)
    print("Loaders ready both")
    epochs = model_args.epochs
    scheduler = CustomLRScheduler(
    optimizer,
    warmup_iters=model_args.warmup_iters,
    lr_decay_iters=model_args.lr_decay_iters,
    min_lr=model_args.min_lr,
    max_lr=model_args.max_lr
)
    # train_step_iterator = range(len(train_dataloader))
    # if device == 0:  # Only create progress bar on rank 0
    #   train_step_iterator = tqdm(train_step_iterator, desc="Training Progress", position=0, leave=True)

        # Print progress on rank 0
    train_loader_length = 0
    train_data_iterator = iter(train_dataloader)
    val_data_iterator = iter(val_loader)
    token_count = 0
 
    for step in tqdm(range(model_args.total_iters)):
 
        print("Step : ", step, "/", model_args.total_iters)
        # print('Total batches: ', len(train_dataloader))
        print("Total gradient accumulation steps: ", model_args.gradient_accumulation_steps)
        print("Total tokens processed: ", token_count)
        
       
        if (step  % model_args.eval_iters == 0 and step != 0) or step == model_args.total_iters - 1:
            losses = estimate_loss( val_loader, val_data_iterator, model_args.device)
            # avg_train_loss = losses['train']
            avg_val_loss = losses['val']
           
            print(f"[GPU {model_args.device}] | Step: {step} / {model_args.total_iters} | Val Loss: {losses['val']:.4f}")
            
            avg_val_loss = torch.Tensor([losses['val']]).to(model_args.device)
            # torch.distributed.reduce(avg_train_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
            # torch.distributed.reduce(avg_val_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
            
            # if device == 'cuda:0':
                # all_gpus_avg_train_loss = avg_train_loss / world_size
                # print(f"All_GPUs_Train_losses: {all_gpus_avg_train_loss.item():.4f}")
            # all_gpus_avg_val_loss = avg_val_loss / world_size
            print(f"Val Loss: {avg_val_loss.item():.4f}")
            
      
            
            perplexity = torch.exp(torch.tensor(avg_val_loss.item()))  # Calculate perplexity

            # if device == 0:
            wandb.log({
                    "All GPU Val_Loss": avg_val_loss.item(),
                    "Val Perplexity": perplexity.item(),
                    "Total Tokens Processed": token_count,
                    "Step": step,
                })
            print(f"Step: {step} | All GPU Val Loss: {avg_val_loss.item():.4f} | Perplexity: {perplexity.item():.4f} | Tokens: {token_count}")
            
            


        if step % model_args.save_checkpoint_iter == 0 and step != 0:
            print(f"Saving the model checkpoint for step: {step}")
            _save_snapshot(model, optimizer, None, None, step)
        
        accumulated_loss = 0.0
        
        
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(model_args.gradient_accumulation_steps):
            try:
                batch = next(train_data_iterator)
            except StopIteration:
                train_data_iterator = iter(train_dataloader)
                batch = next(train_data_iterator)

            idx = batch['input_ids'].to(model_args.device)

            targets = batch['labels'].to(model_args.device)
            token_count += (idx.numel())
            # with torch.autocast(device_type=model_args.device, dtype=torch.bfloat16):
            if(model_args.use_liger):
                loss = model(idx, actual_labels = targets)
            else:
                logits = model(idx)
                batch_size, block_size, embeddings_dims = logits.shape
                # print(logits.shape)   
                # print(targets)
                logits = logits.view(batch_size*block_size, embeddings_dims)
                # print("OK")
                targets = targets.view(batch_size * block_size)
                # print("OK2")
                loss = nn.functional.cross_entropy(logits, targets, ignore_index=tokenizer.pad_token_id)
            
            loss = loss / model_args.gradient_accumulation_steps #IDK why div is done here specifically? Maybe think of it in terms of a very big batch being processed and there is need for equal important of each mini batch for the overall big batch
            accumulated_loss += loss.detach()
            # scaler.scale(loss).backward()
            loss.backward() 
        # model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1) # so that we dont synchronize the gradient everytime across the GPU devices
           
        # Check for unused parameters
    # del logits, targets, loss
    

        unused_params = find_unused_parameters(model)
        if unused_params:
            print(f"Unused parameters: {unused_params}")
    # break

        # if(device == 0):
            # if(micro_step % 10 == 0):
        #     if(step == train_loader_length):
        #       break
                
            # print("Micro Batch : ", micro_step)
        print("Step : ", step, "/", model_args.total_iters)
            # print('Total batches: ', len(train_dataloader))
            # print("Total gradient accumulation steps: ", model_args.gradient_accumulation_steps)
        print("Total tokens processed: ", token_count)
        # count += 1
       
        # lr = cyclical_lr(step)
        # for params in optimizer.param_groups:
        #     params['lr'] = lr
            
        
        
        # Compute gradient norms before clipping
        if(model_args.clip != 0.0):
            
            # scaler.unscale_(optimizer) #To avoid underflow
            total_norm_before = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2
            )

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=model_args.clip)

            # Compute gradient norms after clipping
            total_norm_after = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2
            )
            
            if(step !=0):
                print(f"Gradient Norm Before Clipping: {total_norm_before.item():.4f}")
                print(f"Gradient Norm After Clipping: {total_norm_after.item():.4f}")
        
        optimizer.step()
        scheduler.step()
        # torch.cuda.empty_cache()
        torch.cuda.synchronize() 
        # accumulated_loss = loss
        
        # torch.distributed.reduce(loss, dst=0, op=torch.distributed.ReduceOp.SUM)
        # loss /= world_size
        perplexity = torch.exp(torch.tensor(accumulated_loss.item()))  # Calculate perplexity
        # if(device == 0):
        wandb.log({
                    "Learning Rate": scheduler.get_last_lr()[0],
                    "Train_Loss": accumulated_loss.item(),
                    # "Train loss": loss.item(),
                    "Train Perplexity": perplexity.item(),
                    "Total Tokens Processed": token_count,
                    "Step": step,
                    "Gradient Norm": total_norm_before.item(),
                    # "Epoch": epoch
                    
        })

        accumulated_loss = 0.0
        if step % 200 == 0:
            count = 1
            while(count):  
                prompt = "Once upon a time "
                generated_text = topk_sampling(model, prompt, tokenizer=tokenizer, max_length=model_args.block_size, top_k=50, temperature=1.0, device=model_args.device)
    
     
                print(f" Step: {step} | Generated Text: {generated_text}")

                save_to_file(step, generated_text)
                count -= 1
        
    

        # break
        # if step % 5 == 0:
        #     torch.cuda.empty_cache()
    # Cleanup
    # if device == 0:
        # writer.close()
    wandb.finish()
    cleanup()

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)  # Enable FlashAttention

    torch.set_float32_matmul_precision('high')

    # Initialize model args from command line
    model_args = create_model_args()
    
    # scaler = torch.amp.GradScaler(enabled=(model_args.dtype == 'float16'))

    world_size = torch.cuda.device_count()
    print(f"World size: {world_size}")
    
    # Pass model_args to train function
    train(model_args)
