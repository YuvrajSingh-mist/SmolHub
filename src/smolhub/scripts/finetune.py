import torch
from smolhub.helper.save_model import SaveModel
from smolhub.helper.dataset.load_config import Config

from torch.cuda.amp import GradScaler
from smolhub.helper.count_parameters import count_parameters
from tqdm import tqdm
from smolhub.helper.visualize import Visualizer
# from tests.print_model_details import print_model




class SFTTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, optimizer, loss_fn, tokenizer, scheduler=None):

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.config = Config().get_config()
        self.visualizer = Visualizer()  

        # self.device = 'cpu'
        self.model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs =  self.config["Model"]["epochs"]
        # self.model.train()
        self.eval_iters =  self.config["Model"]["eval_iters"]
        
        #None cus there is loss mask to multiply it with
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        
        #Getting model details and showing to the user
        # print_model(self.model, self.train_dataloader)
      
        print("Total trainable parameters:", count_parameters(self.model) ," which is: " , (count_parameters(self.model) / 163037184 )*100 , "%\ of" , 163037184 , "trainable params")
        print("\n")

        # Mixed Precision Training
        self.scaler = GradScaler(enabled=( self.config["MAP"]["use_float16"]))
        
        #Using torch.compile for efficient training
        if( self.config["Optimizations"]["use_compile"]):
            self.model = torch.compile(self.model)
        
        #load scheduler
        self.scheduler = scheduler
        
        self.save_model = SaveModel(self.config['Model']['save_model_path'])
            
    @torch.inference_mode()
    def evaluate(self):
        
        out = {}
        self.model.eval()

        total_loss = 0.0
        total_batches = 0
        
        
        for split in ['val']:
            eval_steps = None
            val_data_iterator = iter(self.val_dataloader)
            eval_steps = self.config["Model"]["eval_steps"]
            
            if eval_steps is None or eval_steps == 0:
                # If eval_steps is not set, use the length of the dataloader
                eval_steps = len(self.val_dataloader)
                
            for k in range(eval_steps):  # Updated to use eval_steps instead of len(self.val_dataloader)
                
                try:
                    batch = next(val_data_iterator)
                except StopIteration:
                    print(f"Resetting val iterator for step {k}")
                    val_data_iterator = iter(self.val_dataloader)
                    batch = next(val_data_iterator)
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if  self.config["MAP"]["use_bfloat16"] or  self.config["MAP"]["use_float16"] else torch.float32):
                    # batch= next(self.train_dataloader)
                    idx = batch['input_ids'].to(self.device)
                    targets = batch['labels'].to(self.device)
                    loss_mask = batch['loss_mask'].to(self.device)
                    logits = self.model(idx).logits
                    batch_size, block_size, embeddings_dims = logits.shape
                    logits = logits.view(batch_size*block_size, embeddings_dims)
                    targets = targets.view(batch_size * block_size)
                    loss = self.loss_fn(logits, targets)
                loss = loss * loss_mask.view(-1)
                loss = loss.mean()
            
                total_loss += loss.item()
                total_batches += 1
            
        out[split] = total_loss / total_batches  # Average loss
        

        self.model.train()
        return out

    def train(self):
        
        # self.model.train()
        for epoch in range(self.epochs):
            
            train_iterator = iter(self.train_dataloader)
            for step in tqdm(range(len(self.train_dataloader))):
                
                self.total_steps = len(self.train_dataloader) * self.epochs
                if (step  % self.eval_iters == 0 and step != 0) or step == self.total_steps - 1:
                    losses = self.evaluate()
                    print(f"epoch {epoch}, step {step}: val loss {losses['val']:.4f}")
                    self.visualizer.log({
                        "val_loss": losses['val'],
                        
                    })
                    
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    print(f"Resetting train iterator for epoch {epoch + 1}: step {step}")
                    train_iterator = iter(self.train_dataloader)
                    batch = next(train_iterator)
                
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if  self.config["MAP"]["use_bfloat16"] or  self.config["MAP"]["use_float16"] else torch.float16):
                    # batch= next(self.train_dataloader)
                    idx = batch['input_ids'].to(self.device)
                    targets = batch['labels'].to(self.device)
                    loss_mask = batch['loss_mask'].to(self.device)
                    
                    logits = self.model(idx).logits
                    # print(logits)
                    batch_size, block_size, embeddings_dims = logits.shape
                    logits = logits.view(batch_size*block_size, embeddings_dims)
                    targets = targets.view(batch_size * block_size)
                    loss = self.loss_fn(logits, targets)
                    # print(loss_mask)
                    # print(idx)
                    # print(targets)
                    loss = loss * loss_mask.view(-1)
                    loss = loss.mean()
                
                print('Train: ', loss.item())
                loss.requires_grad_(True)
                if  self.config["MAP"]["use_float16"]:
                    self.scaler.scale(loss).backward()
                    # Unscale gradients and optionally clip them
                    # self.scaler.unscale_(self.optimizer)
                  
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                   
                    self.optimizer.step()
                    # self.optimizer.step()
                self.scheduler.step()

                self.visualizer.log({
                    "epoch": epoch,
                    "step": step,
                    "train_loss": loss.item(),
                    "lr": self.scheduler.get_last_lr()[0],
                    # 'grad_norm': total_norm_before.item(),
                })
                
        print("Training completed!")
        self.visualizer.close()
        
        # Save the model
        self.save_model.save(self.model, self.tokenizer)
        print("Model saved successfully!")