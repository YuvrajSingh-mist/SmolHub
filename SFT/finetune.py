import torch 
from load_config import Config
from tests import print_model_details

config = Config().get_config()

class SFTTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, loss_fn):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = config["Model"]["epochs"]
        
        self.model.train()

        #Getting model details and showing to the user
        print_model_details(self.model, self.train_dataloader)
        print("Total trainable parameters:", count_parameters(self.model) , " which is: " , (count_parameters(self.model) / 163037184 )*100 , "%\ of" , 163037184 , "trainable params")
        print("\n")


    def evaluate(self):
        # Optimizer setup and scheduler steup

        optimizer = self.optimizer
        # optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, weight_decay=weight_decay_optim)


        @torch.inference_mode()
        def estimate_loss():
            out = {}
            lora_model.eval()
            for split in ['train', 'val']:
                losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    idx, targets = get_batch(split=split)
                    logits = lora_model(idx).logits
                    batch_size, block_size, embeddings_dims = logits.shape
                    logits = logits.view(batch_size*block_size, embeddings_dims) # Total tokens(words) => batch_size * block_size
                    targets = targets.view(batch_size * block_size)
                    loss = nn.functional.cross_entropy(logits, targets)
                    losses[k] = loss.item()
                out[split] = losses.mean()
            lora_model.train()
            return out

    def train(self):
        
