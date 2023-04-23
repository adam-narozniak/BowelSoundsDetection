"""Module used for data loading for the visualization purposes."""
import pathlib

import numpy as np
import pandas as pd


def load_labels_from_txt_files(files_dir: pathlib.Path = pathlib.Path("data/raw/")) -> list[pd.DataFrame]:
    # Find all .txt files in the directory
    txt_file_paths = files_dir.glob("*.txt")
    dataframes = []

    for file_path in txt_file_paths:
        print(f"Processing {file_path}")
        dataframes.append(load_labels_from_single_txt_file(file_path))
    return dataframes


def load_labels_from_single_txt_file(file_path) -> pd.DataFrame:
    labels_dict = {"start": [],
                   "end": [],
                   "freq_beg": [],
                   "freq_end": []
                   }
    # label_id = 0
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            split_line = line.split()
            # print(idx, split_line)
            if idx % 2 == 0:
                labels_dict["start"].append(split_line[0])
                labels_dict["end"].append(split_line[1])
            else:
                labels_dict["freq_beg"].append(split_line[1])
                labels_dict["freq_end"].append(split_line[2])
                # label_id += 1
    index = pd.MultiIndex.from_product([[file_path.name], list(range(len(labels_dict["start"])))],
                                       names=("file_name", "file_index"))
    return pd.DataFrame(labels_dict, index=index).astype(np.float)
