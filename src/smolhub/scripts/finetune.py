import torch
from smolhub.helper.dataset.load_config import Config

from torch.cuda.amp import GradScaler
from smolhub.helper.count_parameters import count_parameters
from tqdm import tqdm
from smolhub.helper.visualize import Visualizer
# from tests.print_model_details import print_model




class SFTTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, optimizer, loss_fn, scheduler=None):

        self.model = model
        self.train_dataloader = iter(train_dataloader)
        self.val_dataloader = iter(val_dataloader)
        self.test_dataloader = iter(test_dataloader)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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

        if( self.config["Optimizations"]["use_compile"]):
            self.model = torch.compile(self.model)
        #load scheduler
        self.scheduler = scheduler


            
    @torch.inference_mode()
    def evaluate(self):
        
        out = {}
        self.model.eval()

        for split in ['val']:
            losses = torch.zeros(eval_iters := len(self.val_dataloader), device=self.device)
            for k in range(len(self.val_dataloader)):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if  self.config["MAP"]["use_bfloat16"] or  self.config["MAP"]["use_float16"] else torch.float32):
                    batch= next(self.train_dataloader)
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
                # print('Val: ', loss.item())
            out[split] = losses.mean()

        self.model.train()
        return out

    def train(self):
        
        # self.model.train()
        for epoch in range(self.epochs):
            for step in tqdm(range(len(self.train_dataloader))):

                self.total_steps = len(self.train_dataloader) * self.epochs
                if (step  % self.eval_iters == 0 and step != 0) or step == self.total_steps - 1:
                    losses = self.evaluate()
                    print(f"epoch {epoch}, step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                    self.visualizer.log({
                        "val_loss": losses['val'],
                        
                    })
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if  self.config["MAP"]["use_bfloat16"] or  self.config["MAP"]["use_float16"] else torch.float16):
                    batch= next(self.train_dataloader)
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