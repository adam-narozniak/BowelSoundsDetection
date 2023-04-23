import os
import pathlib
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import wandb
from hyperopt import STATUS_OK, Trials, fmin, tpe
from loguru import logger
from wandb.util import generate_id

from bowel.controller import Controller
from bowel.data.data_loader import DataLoader
from bowel.data.label_creator import LabelCreator
from bowel.utils.check_gpu import log_tensorflow_gpu_availability
from bowel.utils.experiments import preprocess_trials
from bowel.utils.io import silence_librosa_userwarnings, load_divsion_file
from bowel.utils.reproducibility_utils import setup_seed
from bowel.utils.wandb_utils import set_custom_wandb_summary
from configs.experiments_config import experiments_params

NOW_STR = datetime.now().strftime("%Y%m%d_%H%M%S")

logger.add(f"./data/experiments_{NOW_STR}.log")

current_experiment = 0
max_evals = 200


def cross_val_wrapper(params, data):
    """Wrapp cross validation to meet hyperopt requirements."""
    global current_experiment
    global max_evals
    logger.info(f"Experiment {current_experiment}/{max_evals}")
    wandb_group_id = generate_id(length=16)
    logger.info(f"WANDB project: {os.environ.get('WANDB_PROJECT')}, group_id: {wandb_group_id}")
    # params["hop_length"] = int(params["frame_length"] / params["frame_to_hop_len_ratio"])
    floats_to_ints = ["n_mfcc", "max_freq"]
    for fti in floats_to_ints:
        params[fti] = int(params[fti])
    logger.info(f"experiment params:")
    logger.info(f"{params}")
    DATA_DIR = pathlib.Path("./data/processed/")
    # path to the file specifying samples names and division in folds
    DIVISION_FILE_PATH = pathlib.Path("./data/processed/files.csv")
    division_data = load_divsion_file(DIVISION_FILE_PATH)
    label_creator = LabelCreator(DATA_DIR, division_data, params)
    labels = label_creator.labels
    controller = Controller(params, params, data, labels)
    metrics = controller.crossval(wandb_group_id).copy()
    metrics["loss"] = - metrics["f1"]
    metrics["status"] = STATUS_OK
    current_experiment += 1
    set_custom_wandb_summary(controller.train_evaluation_score, prefix="avg_train_evaluation/")
    set_custom_wandb_summary(controller.test_evaluation_score, prefix="avg_test_evaluation/")
    wandb.finish()
    del controller.model  # possible data leak on tf + hyperopt side
    del controller  # possible data leak on tf + hyperopt side
    return metrics


def start_experiments(max_evals):
    """Start experiments using hyperopt TPE method and configuration file."""
    setup_seed()
    silence_librosa_userwarnings()
    log_tensorflow_gpu_availability()
    DATA_DIR = pathlib.Path("./data/processed/")
    DIVISION_FILE_PATH = pathlib.Path("./data/processed/files.csv")

    # data loading
    division_data = load_divsion_file(DIVISION_FILE_PATH)

    # initializations
    data_loader = DataLoader(DATA_DIR, division_data)
    data = data_loader.load_data()
    # hold results in Trials
    trials = Trials()
    best = fmin(partial(cross_val_wrapper, data=data),
                experiments_params,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                rstate=np.random.default_rng(0))
    logger.info(f"experiments - best: {best}")
    experiments_results = pd.DataFrame(
        pd.DataFrame(trials.trials)["misc"].map(lambda x: x["idxs"]).transform(preprocess_trials).tolist())
    for k, indices in trials.idxs.items():
        experiments_results.loc[indices, k] = trials.vals[k]
    experiments_results = pd.concat([experiments_results, pd.DataFrame(trials.results)], axis=1)
    logger.info("results")
    logger.info(experiments_results)
    experiments_results.to_csv(f"./data/experiments_results_{NOW_STR}.csv")


if __name__ == "__main__":
    os.environ['WANDB_MODE'] = 'online'
    os.environ["WANDB_PROJECT"] = f"bowel-experiments-{NOW_STR}"
    start_experiments(max_evals)
