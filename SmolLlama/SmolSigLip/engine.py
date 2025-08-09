"""
Contains functions for training and testing a PyTorch model.
"""
import torch
import wandb

from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional

log_sigmoid = torch.nn.LogSigmoid()

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module,
               gradient_accumulation_steps: int,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               epoch: int = 0,
               validate_every_n_steps: int = 500,
               test_dataloader: Optional[torch.utils.data.DataLoader] = None,
               step_counter: Optional[dict] = None,
               lr_scheduler = None) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Use step counter from main train function if provided
    if step_counter is None:
        step_counter = {'count': 0}
    
    # Create progress bar for training batches
    train_pbar = tqdm(enumerate(dataloader), 
                     total=len(dataloader),
                     desc=f"🚀 Epoch {epoch+1:02d} Training",
                     leave=False,
                     dynamic_ncols=True,
                     colour='green')
    
    # Loop through data loader data batches
    for batch_idx, batch in train_pbar:
        # Send data to target device
        batch['text']['input_ids'] = batch['text']['input_ids'].to(device).squeeze(1)
        batch['text']['attention_mask'] = batch['text']['attention_mask'].to(device).squeeze(1)
        batch['image'] = batch['image'].to(device)
        
        batch_size = batch['image'].shape[0]
        
        # Zero gradients once per gradient accumulation cycle
        optimizer.zero_grad()
        
        accumulated_loss = 0.0

        for micro_step in range(gradient_accumulation_steps):
            # labels = torch.arange(batch['text']['input_ids'].shape[0], device=device)
            labels = 2 * torch.eye(batch_size, device=device) - torch.ones(batch_size, device=device)
            
            # 1. Forward pass
            y_pred = model(batch)

            # 2. Calculate loss and scale by gradient accumulation steps
            # loss_i = torch.nn.functional.cross_entropy(y_pred, labels)
            # loss_t = torch.nn.functional.cross_entropy(y_pred.T, labels.T)
            loss = -torch.sum(log_sigmoid(y_pred * labels)) / batch_size
            
            # Scale loss by gradient accumulation steps for proper averaging
            loss = loss / gradient_accumulation_steps
            accumulated_loss += loss.item()

            # 3. Backward pass (gradients accumulate across micro-steps)
            loss.backward()

            if micro_step % 10 == 0:
                print(f"micro step: {micro_step}/{gradient_accumulation_steps} | loss: {loss.item():.6f}")

        # 4. Optimizer step after all micro-steps
        optimizer.step()
        
        # 5. Learning rate scheduler step (if provided)
        if lr_scheduler is not None:
            current_lr = lr_scheduler.step()
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        # Log step-level metrics to wandb
        step_counter['count'] += 1
        
        # Update progress bar with current metrics
        batch_size = batch['image'].shape[0]
        
        # Update tqdm progress bar with detailed info
        train_pbar.set_postfix({
            'Loss': f'{accumulated_loss:.4f}',
            'LR': f'{current_lr:.2e}',
            'BS': batch_size,
            'Step': step_counter['count']
        })
        
        # Simplified logging - only log to wandb without explicit step parameter
        # This allows wandb to handle step incrementing automatically
        wandb.log({
            "train/loss": accumulated_loss,
            "train/learning_rate": current_lr,
            "global_epoch": epoch + 1,
            "dataloader_step": step_counter['count']
        })

        # Run validation every N steps
        if test_dataloader is not None and step_counter['count'] % validate_every_n_steps == 0 and step_counter['count'] > 0:
            print(f"\n🔍 --- Running validation at step {step_counter['count']} ---")
            val_loss, _ = test_step(model, test_dataloader, loss_fn, device, epoch)
            print(f"✅ Validation Loss at step {step_counter['count']}: {val_loss:.6f}")
            
            # Log validation results without explicit step parameter
            wandb.log({
                "val/loss": val_loss,
                "val/dataloader_step": step_counter['count'],
            })
            
            # Switch back to training mode
            model.train()
            print("🔄 --- Validation complete, resuming training ---\n")

        # Save model checkpoint every 500 steps
        if step_counter['count'] % 500 == 0 and step_counter['count'] > 0:
            checkpoint_path = f"checkpoint_step_{step_counter['count']}.pt"
            print(f"💾 --- Saving model checkpoint at step {step_counter['count']} to {checkpoint_path} ---")
            
            # Save model state dict
            torch.save({
                'step': step_counter['count'],
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            
            # Log checkpoint info without explicit step parameter
            wandb.log({
                "checkpoint/dataloader_step": step_counter['count'],
                "checkpoint/saved": True,
            })
            
            print(f"✅ Checkpoint saved successfully at step {step_counter['count']}")
            print("🔄 --- Continuing training ---\n")

        # Calculate and accumulate accuracy metric across all batches
        # y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        # train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per epoch 
    train_loss = train_loss / len(dataloader)
    # train_acc = train_acc / len(dataloader)
    
    # Convert to float to avoid tensor/gradient issues
    train_loss_float = float(train_loss) if hasattr(train_loss, 'item') else train_loss
    return train_loss_float, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device,
              epoch: int = 0) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    batch_count = 0

    # Create progress bar for validation batches
    val_pbar = tqdm(enumerate(dataloader), 
                   total=len(dataloader),
                   desc=f"🔍 Epoch {epoch+1:02d} Validation",
                   leave=False,
                   dynamic_ncols=True,
                   colour='blue')

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch_idx, batch in val_pbar:
         # Send data to target device
          batch['text']['input_ids'] = batch['text']['input_ids'].to(device).squeeze(1)
          batch['text']['attention_mask'] = batch['text']['attention_mask'].to(device).squeeze(1)
          batch['image'] = batch['image'].to(device)
          
          batch_size = batch['image'].shape[0]
          # labels = torch.arange(batch['text']['input_ids'].shape[0], device=device)
          labels = 2 * torch.eye(batch_size, device=device) - torch.ones(batch_size, device=device)
          
          # 1. Forward pass
          y_pred = model(batch)

          # 2. Calculate  and accumulate loss
          # loss_i = torch.nn.functional.cross_entropy(y_pred, labels)
          # loss_t = torch.nn.functional.cross_entropy(y_pred.T, labels.T)
          loss = -torch.sum(log_sigmoid(y_pred * labels)) / batch_size
          # loss_tot = (loss_i + loss_t) / 2.0
          test_loss += loss.detach()  # Detach from computation graph
          batch_count += 1

          # Update validation progress bar
          val_pbar.set_postfix({
              'Val Loss': f'{loss.item():.4f}',
              'Batch': f'{batch_idx+1}/{len(dataloader)}',
              'Avg Loss': f'{(test_loss/batch_count):.4f}'
          })

          # Log validation batch metrics occasionally (without step to avoid conflicts)
          if batch_idx % 50 == 0:  # Log every 50 batches to avoid too much logging
              print(f"Validation batch {batch_idx}: Loss={loss.item():.6f}")
              # Note: Not logging individual validation batches to wandb to keep it simple


          # # 3. Optimizer zero grad
          # optimizer.zero_grad()

          # # 4. Loss backward
          # loss.backward()

          # # 5. Optimizer step
          # optimizer.step()

          # Calculate and accumulate accuracy metric across all batches
          # y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
          # train_acc += (y_pred_class == y).sum().item()/len(y_pred)

        # Adjust metrics to get average loss and accuracy per batch 
        test_loss = test_loss / len(dataloader)
        # train_acc = train_acc / len(dataloader)
        
        # Convert to float to avoid tensor/gradient issues
        test_loss_float = float(test_loss) if hasattr(test_loss, 'item') else test_loss
        return test_loss_float, test_acc

def train(model, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          gradient_accumulation_steps: int,
          writer: None,
          validate_every_n_steps: int = 500,
          lr_scheduler = None) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    validate_every_n_steps: Run validation every N training steps (default: 500).

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }
    
    # Make sure model on target device
    model.to(device)
    
    # Initialize step counter for local step tracking
    step_counter = {'count': 0}

    # Create main progress bar for epochs
    epoch_pbar = tqdm(range(epochs), 
                     desc="🎯 Training Progress",
                     unit="epoch",
                     colour='magenta')

    # Loop through training and testing steps for a number of epochs
    for epoch in epoch_pbar:
        print(f"\n=== EPOCH {epoch + 1}/{epochs} ===")
        
        # Update main epoch progress bar
        epoch_pbar.set_description(f"🎯 Epoch {epoch+1}/{epochs}")
        
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device,
                                          gradient_accumulation_steps=gradient_accumulation_steps,
                                          epoch=epoch,
                                          validate_every_n_steps=validate_every_n_steps,
                                          test_dataloader=test_dataloader,
                                          step_counter=step_counter,
                                          lr_scheduler=lr_scheduler)
        
        print(f"\n--- End of epoch validation ---")
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device,
          epoch=epoch)

        # Update epoch progress bar with current metrics
        # Values are already floats from train_step and test_step
        train_loss_display = train_loss
        test_loss_display = test_loss
        
        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss_display:.4f}',
            'Val Loss': f'{test_loss_display:.4f}',
            'Ratio': f'{test_loss_display/train_loss_display:.3f}' if train_loss_display > 0 else 'N/A'
        })

        # Print out what's happening
        print(
          f"\n🎯 Epoch {epoch+1}/{epochs} Summary:\n"
          f"  📈 Train Loss: {train_loss_display:.6f}\n"
          f"  📊 Test Loss:  {test_loss_display:.6f}\n"
          f"  📏 Loss Ratio: {test_loss_display/train_loss_display:.3f}\n"
          f"  {'📉 Improving' if epoch > 0 and test_loss_display < results['test_loss'][-1] else '📈 Loss increased' if epoch > 0 else '🆕 First epoch'}\n"
        )

        # Log epoch-level metrics to wandb - values are already floats
        train_loss_float = train_loss
        test_loss_float = test_loss
        
        # Log epoch metrics without explicit step parameter
        wandb.log({
            "epoch/train_loss": train_loss_float,
            "epoch/test_loss": test_loss_float,
            "global_epoch": epoch + 1,
        })

        # Update results dictionary
        results["train_loss"].append(train_loss)
        # results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        # results["test_acc"].append(test_acc)

        # Legacy writer support (keeping for backward compatibility)
        if writer:
          writer.add_scalars(main_tag="Loss", 
                                  tag_scalar_dict={"train_loss": train_loss,
                                                    "test_loss": test_loss},
                                  global_step=epoch)
          # writer.add_scalars(main_tag="Accuracy", 
          #                           tag_scalar_dict={"train_acc": train_acc,
          #                                             "test_acc": test_acc}, 
          #                           global_step=epoch)

          writer.close()
        
        else:
            pass
          
    # Return the filled results at the end of the epochs
    return results

def initialize_wandb(project_name: str = "SmolSigLip", 
                    config: Optional[dict] = None,
                    run_name: Optional[str] = None) -> None:
    """Initialize Weights & Biases logging.
    
    Args:
        project_name: Name of the wandb project
        config: Configuration dictionary to log
        run_name: Optional name for this run
    """
    wandb.init(
        project=project_name,
        config=config,
        name=run_name,
        save_code=True,
        tags=["siglip", "contrastive", "vision-language"]
    )
    
    print("✅ Wandb initialized successfully!")
    print(f"📊 Project: {project_name}")
    if wandb.run:
        print(f"🏃 Run: {wandb.run.name}")
        print(f"🆔 Run ID: {wandb.run.id}")
        print(f"🌐 Dashboard: {wandb.run.url}")
    
    print("🔧 Setting up custom metrics...")
    # Define simplified metrics without explicit step dependencies
    wandb.define_metric("train/loss")
    wandb.define_metric("train/learning_rate") 
    wandb.define_metric("global_epoch")
    wandb.define_metric("dataloader_step")
    
    # Define validation metrics
    wandb.define_metric("val/loss")
    wandb.define_metric("val/dataloader_step")
    
    # Define checkpoint metrics
    wandb.define_metric("checkpoint/dataloader_step")
    wandb.define_metric("checkpoint/saved")
    
    # Define epoch-level metrics
    wandb.define_metric("epoch/train_loss")
    wandb.define_metric("epoch/test_loss")
    
    # Define model metrics
    wandb.define_metric("model/total_parameters")
    wandb.define_metric("model/trainable_parameters")
    wandb.define_metric("model/non_trainable_parameters")
    
    print("✅ All metrics defined successfully!")
    
def log_model_summary(model: torch.nn.Module, 
                     input_sample: Optional[dict] = None) -> None:
    """Log model architecture and parameters to wandb.
    
    Args:
        model: PyTorch model to log
        input_sample: Sample input for model visualization
    """
    # Log model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    wandb.log({
        "model/total_parameters": total_params,
        "model/trainable_parameters": trainable_params,
        "model/non_trainable_parameters": total_params - trainable_params
    })
    
    # Log model architecture as text
    model_summary = str(model)
    wandb.log({"model/architecture": wandb.Html(f"<pre>{model_summary}</pre>")})
    
    print(f"Model logged to wandb - Total params: {total_params:,}, Trainable: {trainable_params:,}")

def log_gradients(model: torch.nn.Module, step: int) -> None:
    """Log gradient statistics to wandb.
    
    Args:
        model: PyTorch model
        step: Current training step
    """
    total_norm = 0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            
            # Log individual layer gradients (sample a few)
            if 'weight' in name and len(name.split('.')) <= 3:  # Log main layer weights only
                wandb.log({f"gradients/{name}": param_norm.item()}, step=step)
    
    total_norm = total_norm ** (1. / 2)
    wandb.log({
        "gradients/total_norm": total_norm,
        "gradients/param_count": param_count
    }, step=step)
