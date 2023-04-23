import os
import pathlib
import time

import numpy as np
import pandas as pd
import wandb

from bowel.controller import Controller
from bowel.data.data_loader import DataLoader
from bowel.data.label_creator import LabelCreator
from bowel.factories import instantiate_normalizers
from bowel.utils.io import silence_librosa_userwarnings, load_config, load_divsion_file
from bowel.utils.reproducibility_utils import setup_seed


def measure_prediction_performance():
    """Measure performance for various length input size."""
    setup_seed()
    silence_librosa_userwarnings()
    # paths
    CONFIG_DIR = pathlib.Path("./configs/")
    DATA_DIR = pathlib.Path("./data/processed/")
    # path to the file specifying samples names and division in folds
    division_file_path = pathlib.Path("./data/processed/files.csv")
    transform_config_path = CONFIG_DIR / "best_model_transformation.yaml"
    train_config_path = CONFIG_DIR / "best_model_train.yaml"
    transform_config = load_config(transform_config_path)
    train_config = load_config(train_config_path)
    save_path = pathlib.Path("models/best_model")
    division_data = load_divsion_file(division_file_path)
    # make the data seem as if it was from the last fold
    division_data.loc[:, "kfold"] = 5
    # repeat 7 time to go over 10_000 samples
    division_data = pd.DataFrame(np.repeat(division_data.values, 7, axis=0), columns=division_data.columns)
    print(division_data)
    print(division_data.shape)
    data_sizes = [10, 30, 100, 300, 1_000, 3_000]  # number of 2 second files
    results = {}
    for data_size in data_sizes:
        start_time = time.time()
        part_of_division_data = division_data.iloc[:data_size]
        data_loader = DataLoader(DATA_DIR, part_of_division_data)
        label_creator = LabelCreator(DATA_DIR, part_of_division_data, transform_config)
        data = data_loader.load_data()
        labels = label_creator.labels

        controller = Controller(
            transform_config,
            train_config,
            data,
            labels
        )
        controller.audio_normalizer, controller.features_normalizer = instantiate_normalizers(
            controller.transform_config, mode="test", save_path=save_path)
        controller.test_from_saved(save_path=save_path)
        end_time = time.time()
        time_taken = end_time - start_time
        results[data_size] = time_taken
    return results


if __name__ == "__main__":
    os.environ['WANDB_MODE'] = 'disabled'  # do not register the results in wandb
    wandb.init()
    results = measure_prediction_performance()
    results = pd.DataFrame(np.array([list(results.keys()), list(results.values())]).T, columns=["data_size", "time"])
    results.to_csv("./results/performance.csv")
    print(results)
