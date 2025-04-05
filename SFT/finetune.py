import torch
from torch._numpy import dtype 
from load_config import Config
from tests import print_model_details
from torch.cuda.amp import GradScaler
from helper.count_parameters import count_parameters
from tqdm import tqdm
from helper.visualize import Visualizer


config = Config().get_config()
visualizer = Visualizer()

class SFTTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler):

        self.model = model
        self.train_dataloader = iter(train_dataloader)
        self.val_dataloader = iter(val_dataloader)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = config["Model"]["epochs"]
        self.use_scheduler = config["Scheduler"]["use_scheduler"]
        self.type = config["Scheduler"]["type"]
        self.model.train()

        #Getting model details and showing to the user
        print_model_details(self.model, self.train_dataloader)
        print("Total trainable parameters:", count_parameters(self.model) , " which is: " , (count_parameters(self.model) / 163037184 )*100 , "%\ of" , 163037184 , "trainable params")
        print("\n")

        # Mixed Precision Training
        self.scaler = GradScaler(enabled=(config["MAP"]["use_bfloat16"] or config["MAP"]["use_float16"]))

        if(config["Optimizations"]["use_compile"]):
            self.model = torch.compile(self.model)
        #load scheduler
        self.scheduler = scheduler

    @torch.inference_mode()
    def evaluate(self):
        
        out = {}
        self.model.eval()

        for split in ['val']:
            losses = torch.zeros(eval_iters)
            for k in range(len(self.val_dataloader)):
                idx, targets = next(self.val_dataloader)
                logits = self.model(idx).logits
                batch_size, block_size, embeddings_dims = logits.shape
                logits = logits.view(batch_size*block_size, embeddings_dims) 
                targets = targets.view(batch_size * block_size)
                loss = self.loss_fn(logits, targets)
                losses[k] = loss.item()
            
            out[split] = losses.mean()

        self.model.train()
        return out

    def train(self):
        
        self.model.train()
        for epoch in range(self.epochs):
            for step in tqdm(range(len(self.train_dataloader))):

      
                if (step  % self.eval_iters == 0 and step != 0) or step == self.total_steps - 1:
                    losses = self.evaluate()
                    print(f"epoch {epoch}, step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                    visualizer.log({
                        "val_loss": losses['val'],
                        
                    })
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16 if config["MAP"]["use_bfloat16"] else torch.float16):
                    idx, targets = next(self.train_dataloader)
                    logits = self.model(idx).logits
                    batch_size, block_size, embeddings_dims = logits.shape
                    logits = logits.view(batch_size*block_size, embeddings_dims)
                    targets = targets.view(batch_size * block_size)
                    loss = self.loss_fn(logits, targets)

                loss.requires_grad = True

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # self.optimizer.step()
                self.scheduler.step()

                visualizer.log({
                    "epoch": epoch,
                    "step": step,
                    "train_loss": loss.item(),
                    "lr": self.scheduler.get_last_lr()[0]
                })