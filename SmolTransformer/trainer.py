import torch
import torch.nn.functional as F
import torch.optim as optim
import math
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# Local imports
from config import ModelArgs, get_args
from model import Transformer, initialize_tokenizer_in_model
from data import prepare_dataset, load_datasets
from tokenizer import initialize_tokenizer

def _save_snapshot(model, optimizer, scheduler, epoch, step):
    """Save model checkpoint"""
    snapshot = {
        "MODEL_STATE": model.state_dict(),
        "OPTIMIZER_STATE": optimizer.state_dict(),
        "EPOCHS_RUN": epoch,
        "STEP_RUN": step
    }
    torch.save(snapshot, f"checkpoints/snapshot_{step}.pt")
    print(f"Epoch: {epoch} | Step: {step} | Snapshot saved.")

def get_lr(it, model_args):
    """Learning rate scheduler with warmup and cosine decay"""
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
    """Find unused parameters in the model"""
    unused = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused.append(name)
    return unused

def topk_sampling(model, prompt, tokenizer, device, max_length=30, top_k=50, temperature=0.8, repetition_penalty=1.2):
    """Top-k sampling generation"""
    model.eval()
    
    # Tokenize the source text
    input_ids = tokenizer(prompt, add_special_tokens=True, max_length=max_length, padding='max_length', return_tensors="pt")['input_ids'].to(device)

    print("Input IDs: ", input_ids.shape)
    # Create an initial target sequence with just the BOS token
    target_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
    
    src_mask = torch.ones(input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype).to(device)
    src_mask = src_mask.masked_fill(input_ids == tokenizer.pad_token_id, 0)
    
    # First, encode the source sequence - do this only once
    with torch.no_grad():
        encoder_output = model.encoder(input_ids, src_mask)
    
    for i in range(max_length):
        with torch.no_grad():
            # Pass the encoder output and current target sequence to the decoder
            decoder_output = model.decoder(encoder_output, encoder_output, target_ids, src_mask, None)
            
            # Get logits from the final linear layer
            logits = model.linear_layer(model.norm(decoder_output))

            logits = logits[:, -1, :] / temperature
            
            probs = F.softmax(logits, dim=-1)
            
            # Top-k filtering
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            
            # Sample from top-k
            next_token = torch.multinomial(top_k_probs, num_samples=1)
            xcol = torch.gather(top_k_indices, -1, next_token)
            
            target_ids = torch.cat([target_ids, xcol], dim=1)
            
            # Check if we generated an EOS token
            if xcol.item() == tokenizer.eos_token_id:
                break
    
    print("Target IDs: ", target_ids.shape)
    model.train()
    
    # Decode the generated sequence
    generated_text = tokenizer.decode(target_ids[0])
    return generated_text

def beam_search_corrected(
    model,
    prompt: str,
    tokenizer,
    device: str,
    block_size: int,
    beam_width: int = 5,
    max_length: int = 50,
    temperature: float = 1.0
):
    """Beam search generation"""
    model.eval()
    model = model.to(device)

    # Tokenize and prepare source input with padding and mask
    inputs = tokenizer(
        prompt,
        add_special_tokens=True,
        max_length=block_size,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    src_input_ids = inputs["input_ids"].to(device)
    
    src_mask = torch.ones(src_input_ids.shape[0], src_input_ids.shape[1], dtype=torch.long, device=device)
    src_mask = src_mask.masked_fill(src_input_ids == tokenizer.pad_token_id, 0)

    with torch.no_grad():
        # Encoder output is computed once using the properly masked source
        encoder_output = model.encoder(src_input_ids, src_mask)

    # Initialize beams: list of (sequence_tensor, log_probability_score)
    initial_sequence = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device)
    beams = [(initial_sequence, 0.0)]
    completed_sequences = []

    for step in range(max_length):
        if not beams:
            break

        all_next_candidates = []

        for current_seq_tensor, current_score in beams:
            # If sequence already ended with EOS, move to completed and skip expansion
            if current_seq_tensor[0, -1].item() == tokenizer.eos_token_id:
                completed_sequences.append((current_seq_tensor, current_score))
                continue

            # Prepare decoder input (current target sequence)
            decoder_input_ids = current_seq_tensor

            with torch.no_grad():
                decoder_out = model.decoder(
                    encoder_output,
                    encoder_output,
                    decoder_input_ids,
                    src_mask
                )
                # Get logits for the *last* token in the sequence
                logits_last_token = model.linear_layer(model.norm(decoder_out[:, -1, :]))
                logits_last_token = logits_last_token / temperature
                log_probs = F.log_softmax(logits_last_token, dim=-1)

            # Get top k next tokens for *this* beam
            top_k_log_probs, top_k_indices = torch.topk(log_probs.squeeze(0), beam_width, dim=-1)

            for i in range(beam_width):
                next_token_id = top_k_indices[i].item()
                next_token_log_prob = top_k_log_probs[i].item()

                new_seq_tensor = torch.cat(
                    [current_seq_tensor, torch.tensor([[next_token_id]], device=device)],
                    dim=1
                )
                new_score = current_score + next_token_log_prob
                all_next_candidates.append((new_seq_tensor, new_score))

        # Filter out completed sequences from the list of candidates for the next step
        new_beams = []
        temp_completed = []

        all_next_candidates.sort(key=lambda x: x[1], reverse=True)

        for cand_seq, cand_score in all_next_candidates:
            if cand_seq[0, -1].item() == tokenizer.eos_token_id:
                temp_completed.append((cand_seq, cand_score))
            elif len(new_beams) < beam_width:
                new_beams.append((cand_seq, cand_score))

        beams = new_beams
        completed_sequences.extend(temp_completed)

        # Early stopping
        if len(completed_sequences) >= beam_width:
            completed_sequences.sort(key=lambda x: x[1], reverse=True)
            if not beams or (beams and completed_sequences[beam_width-1][1] > beams[0][1]):
                break

    # If no sequences were completed, use the active beams
    if not completed_sequences and beams:
        completed_sequences.extend(beams)
    elif not completed_sequences and not beams:
        model.train()
        return "[Beam Search Error: No sequences generated]"

    # Sort all completed sequences by score (descending)
    completed_sequences.sort(key=lambda x: x[1], reverse=True)

    if not completed_sequences:
        model.train()
        return "[Beam Search Error: No completed sequences found]"

    best_sequence_tensor = completed_sequences[0][0].squeeze(0)
    model.train()
    generated_text = tokenizer.decode(best_sequence_tensor)
    
    return generated_text

@torch.inference_mode()
def estimate_loss(val_loader, val_iterator, device, model, model_args):
    """Estimate validation loss"""
    out = {}
    model.eval()
    
    for split in ['val']:
        print(f"Starting with {split} evaluation...")
        epoch_losses = []
        
        for step in range(model_args.eval_check):  
            try:
                batch = next(val_iterator)
            except StopIteration:
                val_loader_iterator = iter(val_loader)
                batch = next(val_loader_iterator)
            
            total_loss = 0  
            total_batches = 0 
            
            idx = batch['input_ids'].to(device)
            targets_idx = batch['decoder_input_ids'].to(device)
            targets = batch['labels'].to(device)
            
            src_mask = torch.ones(model_args.batch_size, model_args.block_size, dtype=idx.dtype).to(device)
            tgt_mask = torch.ones(model_args.batch_size, model_args.block_size, dtype=idx.dtype).to(device)
            
            # Get tokenizer for pad token
            tokenizer = initialize_tokenizer(model_args.hf_token)
            
            src_mask = src_mask.masked_fill(idx == tokenizer.pad_token_id, 0)
            tgt_mask = tgt_mask.masked_fill(targets_idx == tokenizer.pad_token_id, 0)
            
            with torch.autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu', dtype=torch.float16):
                loss = model(idx, targets_idx, targets, src_mask, tgt_mask)
            
            total_loss += loss.item()
            total_batches += 1
        
            # Compute mean loss for this epoch
            epoch_loss = total_loss / total_batches if total_batches > 0 else 0.0
            epoch_losses.append(epoch_loss)

        # Compute mean loss across all evaluation epochs
        out[split] = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0

    model.train()
    return out

def train():
    """Main training function"""
    # Initialize wandb
    wandb.init(
        project='Translation',
    )
    print("wandb initialized")

    # Get model arguments
    model_args = ModelArgs()
    
    # Initialize tokenizer
    tokenizer = initialize_tokenizer(model_args.hf_token)
    
    # Update vocab sizes based on tokenizer
    model_args.src_vocab_size = len(tokenizer)
    model_args.tgt_vocab_size = len(tokenizer)
    
    # Load datasets
    fw_train = load_datasets(model_args.hf_token)
    try:
        fw_train = fw_train.train_test_split(test_size=0.01)
    except:
        # If split fails, create a simple train/test structure
        fw_train = {'train': fw_train, 'test': fw_train}
    print(fw_train)

    # Initialize model
    model = Transformer(
        src_vocab_size=model_args.src_vocab_size,
        tgt_vocab_size=model_args.tgt_vocab_size,
        use_liger=model_args.use_liger
    )
    
    print(f"Model on device {model_args.device} is ready")

    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=model_args.max_lr,
        betas=(model_args.beta_1, model_args.beta_2),
        weight_decay=model_args.weight_decay_optim,
        eps=model_args.eps,
    )
   
    # Compile model for optimization
    model = torch.compile(model)
    model = model.to(model_args.device)

    # Setup data loaders
    train_dataloader = prepare_dataset('train', model_args.device, model_args.batch_size, model_args, tokenizer, fw_train)
    val_loader = prepare_dataset('val', model_args.device, model_args.batch_size, model_args, tokenizer, fw_train)
    
    print("Loaders ready both")

    # Setup training variables
    train_data_iterator = iter(train_dataloader)
    val_data_iterator = iter(val_loader)
    token_count = 0
    
    # Setup mixed precision training
    scaler = GradScaler(enabled=True)
    
    # Calculate total batches
    batches_per_epoch = len(train_dataloader)
    total_batches = batches_per_epoch * model_args.epochs
    
    print(f"Training info: {model_args.epochs} epochs, {batches_per_epoch} batches per epoch, {total_batches} total batches")
    
    model.train()
    
    for step in tqdm(range(model_args.total_iters)):
        # Evaluate periodically
        if (step % model_args.eval_iters == 0) or step == model_args.total_iters - 1:
            losses = estimate_loss(val_loader, val_data_iterator, model_args.device, model, model_args)
            avg_val_loss = losses['val']
            
            print(f"[GPU {model_args.device}] | Step: {step} / {model_args.total_iters} | Val Loss: {losses['val']:.4f}")
            
            perplexity = torch.exp(torch.tensor(avg_val_loss))
            
            wandb.log({
                "Val_Loss": avg_val_loss,
                "Val Perplexity": perplexity.item(),
                "Total Tokens Processed": token_count,
                "Step": step,
            })
            print(f"Step: {step} | Val Loss: {avg_val_loss:.4f} | Perplexity: {perplexity.item():.4f} | Tokens: {token_count}")

        # Save checkpoint
        if step % model_args.save_checkpoint_iter == 0:
            print(f"Saving the model checkpoint for step: {step}")
            _save_snapshot(model, optimizer, None, None, step)
        
        # Initialize gradient accumulation
        accumulated_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        # Gradient accumulation loop
        for micro_step in range(model_args.gradient_accumulation_steps):
            try:
                micro_batch = next(train_data_iterator)
            except StopIteration:
                train_data_iterator = iter(train_dataloader)
                micro_batch = next(train_data_iterator)

            # Extract data from batch dictionary
            idx = micro_batch['input_ids'].to(model_args.device)
            targets_idx = micro_batch['decoder_input_ids'].to(model_args.device)
            targets = micro_batch['labels'].to(model_args.device)
            
            src_mask = torch.ones(model_args.batch_size, model_args.block_size, dtype=idx.dtype).to(model_args.device)
            tgt_mask = torch.ones(model_args.batch_size, model_args.block_size, dtype=idx.dtype).to(model_args.device)
            
            src_mask = src_mask.masked_fill(idx == tokenizer.pad_token_id, 0)
            tgt_mask = tgt_mask.masked_fill(targets_idx == tokenizer.pad_token_id, 0)
            
            token_count += idx.numel()
            
            with torch.autocast(device_type='cuda' if 'cuda' in str(model_args.device) else 'cpu', dtype=torch.float16):
                loss = model(idx, targets_idx, targets, src_mask, tgt_mask)

            # Scale loss by gradient accumulation steps
            loss = loss / model_args.gradient_accumulation_steps
            accumulated_loss += loss.item()

            print(f" Mini Step: {micro_step} / {model_args.gradient_accumulation_steps} | Loss: {loss.item():.4f}")

            # Backward pass
            scaler.scale(loss).backward()

        # Check for unused parameters
        unused_params = find_unused_parameters(model)
        if unused_params:
            print(f"Unused parameters: {unused_params}")

        # Update learning rate
        lr = get_lr(step, model_args)
        for params in optimizer.param_groups:
            params['lr'] = lr
        
        # Compute gradient norms before clipping
        total_norm_before = 0.0
        if model_args.clip != 0.0:
            scaler.unscale_(optimizer)
            total_norm_before = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2
            )

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=model_args.clip)

            total_norm_after = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2
            )
            
            if step != 0:
                print(f"Gradient Norm Before Clipping: {total_norm_before.item():.4f}")
                print(f"Gradient Norm After Clipping: {total_norm_after.item():.4f}")
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        torch.cuda.empty_cache()
    
        # Calculate metrics
        perplexity = torch.exp(torch.tensor(accumulated_loss))
        
        print("Step : ", step, "/", model_args.total_iters)
        print("Total gradient accumulation steps: ", model_args.gradient_accumulation_steps)
        print("Total tokens processed: ", token_count)
        
        # Log to wandb
        grad_norm_value = total_norm_before.item() if torch.is_tensor(total_norm_before) else total_norm_before
        wandb.log({
                "Learning Rate": lr,
                "Train_Loss": accumulated_loss,
                "Train Perplexity": perplexity.item(),
                "Total Tokens Processed": token_count,
                "Step": step,
                "Gradient Norm": float(grad_norm_value),
                "Gradient Accumulation Steps": model_args.gradient_accumulation_steps,
            })

        # Generate samples periodically
        if step != 0 and step % 500 == 0:
            count = 1
            while count:  
                prompt = ["Hello! Myself an AI Assistant. How are you? ", "My name is Khan ", "How are you? ", "The AI will take our jobs ahhh! "]                   
                
                for prt in prompt:
                    print(f"\nGenerating text for prompt: {prt}")
                    generated_text = topk_sampling(model, prt, tokenizer, str(model_args.device), max_length=model_args.block_size, top_k=100, temperature=0.9, repetition_penalty=1.2)
                    beam_text = beam_search_corrected(model, prt, tokenizer, str(model_args.device), block_size=model_args.block_size, beam_width=5, max_length=model_args.block_size, temperature=0.8)

                    print(f"\nTop-K Generated Text: {generated_text}")
                    print(f"Beam Search Text: {beam_text}\n")
                    
                    with open(f'generated_data/generations_{step}.txt', 'a') as f:
                        f.write(f"------------------------------------------------Step: {step}--------------------------------------------\n\n")
                        f.write(f"Prompt: {prt}\n\n")
                        f.write(f"Top-K Generated Text: {generated_text}\n\n")
                        f.write(f"Beam Search Text: {beam_text}\n\n")
                
                count -= 1
            
    _save_snapshot(model, optimizer, None, None, step)
    wandb.finish()

if __name__ == "__main__":
    train()
