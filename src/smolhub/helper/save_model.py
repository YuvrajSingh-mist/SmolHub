import os
from pathlib import Path
import torch
from smolhub.helper.dataset.load_config import Config


#Saves the trained model

config = Config().get_config()

class SaveModel:
    def __init__(self, savepath):
        # self.model = model
        self.path = Path.cwd() / savepath  # Corrected to use Path.cwd() for proper path handling
        self.path = str(self.path) 
        self.model_name = config["Model"]["saved_model_name"]
        os.makedirs(self.path, exist_ok=True)
        # self.tokenizer = tokenizer
        
    def save(self, model, tokenizer):
        torch.save(model.state_dict(), os.path.join(self.path, self.model_name))
        torch.save(tokenizer, os.path.join(self.path, 'tokenizer.pt'))
        