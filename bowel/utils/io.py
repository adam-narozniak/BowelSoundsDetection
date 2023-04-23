import pathlib
import warnings
from typing import Union

import pandas as pd
import yaml


def load_config(config_path: pathlib.Path) -> dict:
    """Loads yaml config."""
    with open(config_path) as config_file:
        return yaml.safe_load(config_file)


def save_yaml(data: dict[str, float], path: pathlib.Path) -> None:
    with open(path, 'w') as outfile:
        yaml.dump(data, outfile)


def load_divsion_file(path: Union[pathlib.Path, str]) -> pd.DataFrame:
    # rearrange the columns' data order (for desired MultiIndex of that order)
    return pd.read_csv(path).iloc[:, :-1][["kfold", "filename"]]


def silence_librosa_userwarnings() -> None:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
