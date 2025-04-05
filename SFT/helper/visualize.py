import wandb
from load_config import Config

config = Config().get_config()

class Visualizer:
    def __init__(self):
        wandb.init(project=config["wandb"]["project_name"])

    def log(self, metrics: Dict[str, float]):
        wandb.log(metrics)

    def close(self):
        wandb.finish()

    