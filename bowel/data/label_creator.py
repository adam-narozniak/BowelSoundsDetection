import pathlib

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm


class LabelCreator:
    """
    Creates labels based on the csv files specifying periods when the sound is present.

    Each audio file extract has corresponding csv file. All the csv files list is present in the division file.

    Args:
        data_dir: path to the directory containing audio extracts
        division_file_path: path to the csv file containing audio file names and the associated fold
        config: transformation configuration - required to determine the single instance sample length
    """

    def __init__(self, data_dir: pathlib.Path, division_file_path: pathlib.Path, config: dict) -> None:
        self._data_dir = data_dir
        self.division_file_path = division_file_path
        self._config = config
        self._wav_sample_length = None
        self._frame_length = None
        self.n_substamps = None
        self._extract_transformation_info()
        self._annotations = pd.read_csv(self.division_file_path).iloc[:, :-1][["kfold", "filename"]]
        self._labels: np.array = None

    def _extract_transformation_info(self) -> None:
        self._wav_sample_length = self._config["wav_sample_length"]
        self._frame_length = self._config["relative_frame_length"]
        self.n_substamps = int((self._wav_sample_length - self._frame_length) / self._frame_length) + 1

    @property
    def labels(self) -> np.ndarray:
        """Zero or one value for each sample determined based on heuristic in the function `_if_sound_present`."""
        if self._labels is None:
            logger.info("Label creation started")
            self._labels = self._create_labels()
            logger.info("Label creation done")
            return self._labels
        else:
            return self._labels

    def _create_labels(self) -> np.ndarray:
        """
        Creates labels based on the information when the sound occurs for all the files given in the division file.
        """
        filenames = self._annotations["filename"]
        labels = []
        for filename in tqdm(filenames):
            # each file of length of "wav_sample_length" has csv file with annotations
            annotation_file_name = str(filename).replace(".wav", ".csv")
            annotation_file_path = self._data_dir / annotation_file_name
            audio_annotations = pd.read_csv(annotation_file_path)
            sample_labels = self._create_sample_labels(audio_annotations)
            labels.append(sample_labels)
        labels = np.array(labels)
        if self._config["subtimesteps"]:
            pass
        else:
            labels = labels.reshape(-1, 1)
        return labels

    def _create_sample_labels(self, audio_annotations: pd.DataFrame) -> list[int]:
        """Determines the labels for a single file."""
        sample_labels = [0 for _ in range(self.n_substamps)]
        for _, row in audio_annotations.iterrows():
            sound_present_start, sound_present_end = row["start"], row["end"]
            for substamp_id in range(self.n_substamps):
                search_period_start = substamp_id * self._frame_length
                search_period_end = search_period_start + self._frame_length
                label = self._if_sound_present(sound_present_start,
                                               sound_present_end,
                                               search_period_start,
                                               search_period_end)
                previous_label = sample_labels[substamp_id]
                sample_labels[substamp_id] = 0 if (label == 0) and (previous_label == 0) else 1
        return sample_labels

    def _if_sound_present(
            self,
            sound_present_start: float,
            sound_present_end: float,
            search_period_start: float,
            search_period_end: float
    ) -> int:
        """
        Determines the label for single period.

        Sound is treated as present if at least half of the search period contains the sound.

        Returns:
            0 if not present
            1 if present
        """
        sound_present_start = max(0., sound_present_start)
        sound_present_end = min(self._config["wav_sample_length"], sound_present_end)
        # the search period ends before the start of the sound
        if search_period_end < sound_present_start:
            return 0
        # the search period starts after the sound end
        if sound_present_end < search_period_start:
            return 0
        # search period is bigger than the sound (sound is fully present is search period)
        if search_period_start < sound_present_start and search_period_end > sound_present_end:
            sound_len = sound_present_end - sound_present_start
            if sound_len > 0.5 * self._frame_length:
                return 1
            else:
                return 0
        # search period ends before the end of the sound or search period starts after the sound starts
        if search_period_end < sound_present_end or search_period_start > sound_present_start:
            sound_len_in_search_period = min(search_period_end, sound_present_end) - max(search_period_start,
                                                                                         sound_present_start)
            if sound_len_in_search_period >= 0.5 * self._frame_length:
                return 1
            else:
                return 0
