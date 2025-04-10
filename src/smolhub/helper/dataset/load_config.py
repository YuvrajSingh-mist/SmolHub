import yaml


class Config:
    def __init__(self, config_path='/mnt/c/Users/yuvra/OneDrive/Desktop/Work/pytorch/SmolHub/src/smolhub/config/config.yaml'):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def get_config(self):
        return self.config

