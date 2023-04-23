import wandb


def set_custom_wandb_summary(evaluation_scores: dict, prefix: str = ""):
    for k, v in evaluation_scores.items():
        wandb.run.summary[prefix + str(k)] = v
