import wandb
import os


def init_wandb(exp_name=None, config={}):

    exp_name = "exp2"

    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_API_KEY"] = "a7b8fd4b9c4346e595e83c9bec00b807f133d853"
    wandb.init(project="PIE-bottlenose_dolphin", name=exp_name)
    wandb.config = config  # {"learning_rate": 0.001, "epochs": 100, "batch_size": 128}

    return wandb
