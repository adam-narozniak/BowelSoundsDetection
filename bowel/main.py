import argparse
import os
import pathlib

import pandas as pd
import wandb
from wandb.util import generate_id

from bowel import fl
from bowel.controller import Controller
from bowel.data.data_loader import DataLoader
from bowel.data.label_creator import LabelCreator
from bowel.factories import instantiate_normalizers
from bowel.fl.client import Client
from bowel.utils.io import load_config, load_divsion_file, silence_librosa_userwarnings
from bowel.utils.reproducibility_utils import setup_seed
from bowel.utils.wandb_utils import set_custom_wandb_summary


def main(
        mode: str,
        transform_config_name: pathlib.Path,
        train_config_name: pathlib.Path,
        store_test_metrics: bool,
        save_path: pathlib.Path = None,
        log: bool = False,
        wandb_log_name: str = None,
        partition: int = None
) -> None:
    """Starts the training/testing/cross-validation using Controller class, creates required data."""
    setup_seed()
    silence_librosa_userwarnings()
    # paths
    CONFIG_DIR = pathlib.Path("./configs/")
    DATA_DIR = pathlib.Path("./data/processed/")
    # path to the file specifying samples names and division in folds
    DIVISION_FILE_PATH = pathlib.Path("./data/processed/files.csv")

    # data loading
    transform_config_path = CONFIG_DIR / transform_config_name
    train_config_path = CONFIG_DIR / train_config_name
    transform_config = load_config(transform_config_path)
    train_config = load_config(train_config_path)
    full_config = {**train_config, **transform_config}
    division_data = load_divsion_file(DIVISION_FILE_PATH)

    # useful classes initializations
    data_loader = DataLoader(DATA_DIR, division_data)
    label_creator = LabelCreator(DATA_DIR, division_data, transform_config)
    data = data_loader.load_data()
    labels = label_creator.labels

    controller = Controller(
        transform_config,
        train_config,
        data,
        labels
    )
    if not log:
        os.environ["WANDB_MODE"] = "disabled"
    if wandb_log_name != "":
        os.environ["WANDB_NAME"] = wandb_log_name
    if mode == "train":
        wandb.init(config=full_config)
        controller.create_train_test_data()
        controller.instantiate_model(controller.model_type, controller.train_config)
        controller.train()
        for k, v in controller.train_evaluation_score.items():
            wandb.run.summary["train_evaluation/" + str(k)] = v
    elif mode == "train_test":
        wandb.init(config=full_config)
        controller.create_train_test_data()
        controller.train_test()
    elif mode == "crossval":
        wandb_group_id = generate_id(length=16)
        controller.crossval(wandb_group_id)
        set_custom_wandb_summary(controller.train_evaluation_score, prefix="avg_train_evaluation/")
        set_custom_wandb_summary(controller.test_evaluation_score, prefix="avg_test_evaluation/")
    elif mode == "test":
        wandb.init(config=full_config)
        controller.audio_normalizer, controller.features_normalizer = instantiate_normalizers(
            controller.transform_config, mode="test", save_path=save_path)
        controller.test_from_saved(save_path=save_path)
    else:
        raise ValueError(f"The given 'mode' is not supported. Given: {mode}.")
    wandb.finish()
    if (mode == "train" or mode == "train_test") and save_path is not None:
        controller.save_model(save_path)
    if store_test_metrics:
        test_metrics = pd.DataFrame(controller.test_evaluation_score, index=[wandb_log_name])
        model_info = pd.DataFrame.from_dict({a: [b] for a, b in full_config.items()})
        results = pd.concat([model_info, test_metrics.reset_index(), controller.time_book], axis=1)
        results.to_csv("./data/test_metrics.csv", mode="a")


def create_parser() -> argparse.ArgumentParser:
    """Creates parser required to specify params to control the operations of Controller."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str,
                        help="{'train'| 'train_test' | 'crossval' | 'test'}")
    parser.add_argument("--transform_config_name", type=pathlib.Path,
                        help="name of the file to the transformation configuration; "
                             "file must be place in ./configs dir")
    parser.add_argument("--train_config_name", type=pathlib.Path,
                        help="name of the file to the training configuration;"
                             "file must be place in ./configs dir")
    parser.add_argument("-s", "--save_path", type=pathlib.Path, default=None,
                        help="path to a dir to save (or load, depending on the mode) the model; "
                             "model is not saved if left None")
    parser.add_argument("-stm", "--store_test_metrics", action="store_true",
                        help="Stores the metrics on the test sets for a model. "
                             "Saves it to the csv file in data reports/test_metrics.csv"
                             "wandb_log_name is used to distinguish models.")
    parser.add_argument("-l", "--log", action="store_true",
                        help="add this flag if you want to use wandb."
                             "if left empty, the run won't be logged online, instead offline mode will be used")
    parser.add_argument("-n", "--wandb_log_name", type=str, default="",
                        help="name to log the experiment on wandb (ignored if -l flag is not set);"
                             "if not give, a random name will appear")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    os.environ["WANDB_PROJECT"] = "bowel"
    args = vars(parser.parse_args())  # vars to convert namespace to dict
    main(**args)
