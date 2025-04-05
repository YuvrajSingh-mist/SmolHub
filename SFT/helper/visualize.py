import wandb



class Visualizer:
    def __init__(self):
        wandb.init(project="SFTrainer")

    def log(self, metrics: Dict[str, float]):
        wandb.log(metrics)

    def close(self):
        wandb.finish()

    