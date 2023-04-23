import pathlib

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm


class LabelCreator:
    """
    Creates labels based on the csv files specifying periods when the sound is present.

    Each audio file extract has corresponding csv file. All the csv files list is present in the division data.

    Args:
        data_dir: path to the directory containing the audio extracts
        division_data: dataframe containing audio file names and the associated fold
        config: transformation configuration - required to specify the sample length of a single instance
    """

    def __init__(self, data_dir: pathlib.Path, division_data: pd.DataFrame, config: dict) -> None:
        self._data_dir = data_dir
        self._config = config
        self._frame_length_seconds = None
        # the sub prefix is to distinguish one audio extract with a single sample for which the prediction will be made
        self.n_subsamples = None
        self._extract_transformation_info()
        self._division_data = division_data
        self._labels: np.array = None

    def _extract_transformation_info(self) -> None:
        audio_len_in_samples = self._config["sample_length_seconds"] * self._config["sr"]
        frame_length_in_samples = self._config["frame_length"]
        hop_length_in_samples = self._config["hop_length"]
        self._frame_length_seconds = frame_length_in_samples / self._config["sr"]
        # not full subsamples will be discarded further (that's why the int is used)
        padding_length = 2 * int(frame_length_in_samples / 2)
        frames_per_input_data = int(frame_length_in_samples / hop_length_in_samples)
        self.n_subsamples = (1 + int(
            (
                    audio_len_in_samples + padding_length - frame_length_in_samples) / hop_length_in_samples)) // frames_per_input_data

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
        filenames = self._division_data["filename"]
        labels = []
        for filename in tqdm(filenames):
            # each file of length of "sample_length_seconds" has csv file with annotations
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
        sample_labels = [0 for _ in range(self.n_subsamples)]
        for _, row in audio_annotations.iterrows():
            sound_present_start, sound_present_end = row["start"], row["end"]
            for subsample_id in range(self.n_subsamples):
                search_period_start = subsample_id * self._frame_length_seconds
                search_period_end = search_period_start + self._frame_length_seconds
                label = self._if_sound_present(sound_present_start,
                                               sound_present_end,
                                               search_period_start,
                                               search_period_end)
                previous_label = sample_labels[subsample_id]
                sample_labels[subsample_id] = 0 if (label == 0) and (previous_label == 0) else 1
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
        audio_end = self.n_subsamples * self._frame_length_seconds
        sound_present_end = min(audio_end, sound_present_end)
        # the search period ends before the start of the sound
        if search_period_end < sound_present_start:
            return 0
        # the search period starts after the sound end
        if sound_present_end < search_period_start:
            return 0
        # search period is bigger than the sound (sound is fully present is search period)
        if search_period_start < sound_present_start and search_period_end > sound_present_end:
            sound_len = sound_present_end - sound_present_start
            if sound_len > 0.5 * self._frame_length_seconds:
                return 1
            else:
                return 0
        # search period ends before the end of the sound or search period starts after the sound starts
        if search_period_end < sound_present_end or search_period_start > sound_present_start:
            sound_len_in_search_period = min(search_period_end, sound_present_end) - max(search_period_start,
                                                                                         sound_present_start)
            if sound_len_in_search_period >= 0.5 * self._frame_length_seconds:
                return 1
            else:
                return 0
