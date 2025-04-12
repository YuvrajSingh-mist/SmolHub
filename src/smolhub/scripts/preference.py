import torch
from smolhub.helper.save_model import SaveModel
from smolhub.helper.dataset.load_config import Config

from torch.cuda.amp import GradScaler
from smolhub.helper.count_parameters import count_parameters
from tqdm import tqdm
from smolhub.helper.visualize import Visualizer
# from tests.print_model_details import print_model
import torch.nn as nn

class DPO:
        def __init__(self, ref_model, sft_model, device, beta, tokenizer):


            self.ref_model = ref_model
            self.sft_model = sft_model
            self.device=device
            self.beta = beta
            self.tokenizer = tokenizer
            self.ref_model.eval()



        def DPOloss(self, datapoint):

            self.win_prompt = datapoint['chosen']
            self.lose_prompt = datapoint['rejected']

        #Token level aggregation 
            with torch.no_grad():
                self.win_log_ref = torch.nn.functional.log_softmax(self.ref_model(**self.win_prompt).logits, dim=-1)
                self.win_log_ref = torch.gather(self.win_log_ref, -1, self.win_prompt['input_ids'].unsqueeze(-1)).squeeze(-1) #Why gather? Because its not token level stuff we care about but sequence level. Hence, we will sum up the probs of every token to get seq level but we don't want to do it for attention maksed tokens too. Hence we we will use gather() to get the ids and multiply the probs by the masked out tokens indexes.
                # print("Gather: ", self.chosen_log_probs)
                self.win_log_ref = self.win_log_ref * (self.win_prompt['attention_mask'])
                self.win_log_ref = self.win_log_ref.sum(dim=-1)
                
                self.lose_log_ref = torch.nn.functional.log_softmax(self.ref_model(**self.lose_prompt).logits, dim=-1)
                self.lose_log_ref = torch.gather(self.lose_log_ref, -1, self.lose_prompt['input_ids'].unsqueeze(-1)).squeeze(-1) #Why gather? Because its not token level stuff we care about but sequence level. Hence, we will sum up the probs of every token to get seq level but we don't want to do it for attention maksed tokens too. Hence we we will use gather() to get the ids and multiply the probs by the masked out tokens indexes.
                # print("Gather: ", self.chosen_log_probs)
                self.lose_log_ref = self.lose_log_ref * (self.lose_prompt['attention_mask'])
                self.lose_log_ref = self.lose_log_ref.sum(dim=-1)
            
            self.win_log_sft = torch.nn.functional.log_softmax(self.sft_model(**self.win_prompt).logits, dim=-1)
            self.win_log_sft = torch.gather(self.win_log_sft, -1, self.win_prompt['input_ids'].unsqueeze(-1)).squeeze(-1) #Why gather? Because its not token level stuff we care about but sequence level. Hence, we will sum up the probs of every token to get seq level but we don't want to do it for attention maksed tokens too. Hence we we will use gather() to get the ids and multiply the probs by the masked out tokens indexes.
            self.win_log_sft = self.win_log_sft * (self.win_prompt['attention_mask'])
            self.win_log_sft = self.win_log_sft.sum(dim=-1)
            
            self.lose_log_sft = torch.nn.functional.log_softmax(self.sft_model(**self.lose_prompt).logits, dim=-1)
            self.lose_log_sft = torch.gather(self.lose_log_sft, -1, self.lose_prompt['input_ids'].unsqueeze(-1)).squeeze(-1) #Why gather? Because its not token level stuff we care about but sequence level. Hence, we will sum up the probs of every token to get seq level but we don't want to do it for attention maksed tokens too. Hence we we will use gather() to get the ids and multiply the probs by the masked out tokens indexes.
            self.lose_log_sft = self.lose_log_sft * (self.lose_prompt['attention_mask'])
            self.lose_log_sft = self.lose_log_sft.sum(dim=-1)

            self.diff1 = self.win_log_sft - self.win_log_ref
            self.diff2 = self.lose_log_sft - self.lose_log_ref

            self.final = -nn.functional.logsigmoid(self.beta *(self.diff1 - self.diff2)).mean()

            # sft_model.train()
            return self.final
        
class PreferenceAlignmentTrainer:
    def __init__(self, device, ref_model, sft_model, train_dataloader, val_dataloader, test_dataloader, optimizer, loss_fn, tokenizer, scheduler=None):

        self.ref_model = ref_model
        self.sft_model = sft_model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.tokenizer = tokenizer
        self.config = Config().get_config()
        self.visualizer = Visualizer()  

        # self.device = 'cpu'
        self.ref_model.to(self.device)
        self.sft_model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs =  self.config["Model"]["epochs"]
       
        self.eval_iters =  self.config["Model"]["eval_iters"]
        
        #None cus there is loss mask to multiply it with
        if(self.config['Training']['type'] == 'DPO'):
            self.loss_fn = DPO(ref_model, sft_model, device, self.config["Training"]["beta"], tokenizer)
        
        # Mixed Precision Training
        self.scaler = GradScaler(enabled=( self.config["MAP"]["use_float16"]))
        
        #Using torch.compile for efficient training
        if( self.config["Optimizations"]["use_compile"]):
            self.sft_model = torch.compile(self.sft_model)
            self.ref_model = torch.compile(self.ref_model)
        #load scheduler
        self.scheduler = scheduler
        
        self.save_model = SaveModel(self.config['Model']['save_model_path'])
    
    
    


  
    @torch.inference_mode()
    def evaluate(self):
        
        out = {}
        self.sft_model.eval()

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
                 
                    loss = self.loss_fn.DPOloss(batch)
                    
                total_loss += loss.item()
                total_batches += 1
            
        out[split] = total_loss / total_batches  # Average loss
        

        self.sft_model.train()
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
                  
                    loss = self.loss_fn.DPOloss(batch)
                
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
        self.save_model.save(self.sft_model, self.tokenizer)
        print("Model saved successfully!")