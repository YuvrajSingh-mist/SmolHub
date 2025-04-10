from pathlib import Path
import yaml


class Config:
    def __init__(self):
        

        # Look for config in current working directory first
       
        cwd_config = Path.cwd() / "config.yaml"
        print("File found!", cwd_config)
        if cwd_config.exists():
            print("Using existing config file")
            self.config_path = str(cwd_config)
        else:
            print("Config YAML File Not Found...Creating Default Config")
            # Create default config in user's home directory if it doesn't exist
            default_config_dir = Path.home() / ".smolhub"
            default_config_dir.mkdir(exist_ok=True)
            self.config_path = str(default_config_dir / "config.yaml")
            if not Path(self.config_path).exists():
                self._create_default_config()
  
            
        self.config = self.load_config()

    def _create_default_config(self):
        default_config = {
            "Dataset": {
                "use_hf_dataset": True,
                "dataset_path": "stanfordnlp/imdb",
                "max_length": 128,
                "batch_size": 16,
                "num_workers": 4,
                "shuffle": True,
                "drop_last": True,
                "pin_memory": True,
                "persistent_workers": True
            },
            "Model": {
                "model_path": "",
                "epochs": 1,
                "eval_iters": 10,
                "tokenizer": ""
            },
            "MAP": {
                "use_bfloat16": False,
                "use_float16": True
            },
            "Optimizations": {
                "use_compile": False
            },
            "huggingface": {
                "hf_token": "YOUR_HF_TOKEN_HERE"
            }
        }
        
        with open(self.config_path, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)

    def load_config(self):
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def get_config(self):
        return self.config

