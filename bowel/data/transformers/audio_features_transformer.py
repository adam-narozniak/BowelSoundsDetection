import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from bowel.data.transformers.data_transformer import DataTransformer


class AudioFeaturesTransformer(DataTransformer):
    """Creates the following audio features: mfcc, rms."""

    def __init__(self, data: pd.DataFrame, config: dict):
        super().__init__(data, config)
        self._mfccs: np.ndarray
        # self._rmss: np.ndarray

    def _transform(self):
        self._extract_features()
        self._swap_axis()
        # self._concatenate()
        self._transformed = self._mfccs
        self._reshape_data()

    def _extract_features(self):
        sr = self._config["sr"]
        hop_length = self._config['hop_length']
        fmax = self._config['max_freq']
        window = self._config['window_type']
        frame_length = self._config["frame_length"]
        n_mfcc = self._config["n_mfcc"]
        mfccs = []
        # rmss = []
        for audio in tqdm(self._data):
            mfcc = librosa.feature.mfcc(y=audio,
                                        sr=sr,
                                        n_mfcc=n_mfcc,
                                        hop_length=hop_length,
                                        n_fft=frame_length,
                                        window=window,
                                        fmax=fmax)
            # rms = librosa.feature.rms(audio,
            #                           frame_length=frame_length,
            #                           hop_length=hop_length)
            mfccs.append(mfcc)
            # rmss.append(rms)
        self._mfccs = np.asarray(mfccs)
        # self._rmss = np.asarray(rmss)

    def _swap_axis(self):
        """Moves features to the last dimension."""
        self._mfccs = self._mfccs.swapaxes(1, 2)
        # self._rmss = self._rmss.swapaxes(1, 2)

    # def _concatenate(self):
    #     """Concatenates all the features into one np.ndarray."""
    #     self._transformed = np.concatenate([self._mfccs, self._rmss], axis=2)

    def _determine_shape(self):
        """
        Determines the shape based on the audio transformation configuration data (frame_length and hop_length).

        It is needed because some ending of the data requires being removed and data slightly reshaped so a few
        frames (frame_length/hop_length) are grouped together.
        It is based on the information on STFT and total frames calculations from:
        * https://brianmcfee.net/dstbook-site/content/ch09-stft/Framing.html
        * https://github.com/librosa/librosa/issues/1288
        """
        audio_len_in_samples = int(self._config["sample_length_seconds"] * self._config["sr"])
        frame_length_in_samples = self._config["frame_length"]
        hop_length_in_samples = self._config["hop_length"]
        # it's a specific behaviour when centering=True, so the padding of length floor(frame_length/2) is added
        # to both sides
        padding_length = 2 * int(frame_length_in_samples / 2)
        total_n_frames = 1 + int(
            (audio_len_in_samples + padding_length - frame_length_in_samples) / hop_length_in_samples)
        # frame length is dividable by hop length
        frames_per_input_data = int(frame_length_in_samples / hop_length_in_samples)
        n_inputs = total_n_frames // frames_per_input_data
        shape = [self._transformed.shape[0], n_inputs, frames_per_input_data, self._transformed.shape[-1]]
        return shape

    def _reshape_data(self):
        """
        Reshapes the data into the format needed in models, which is:
        (None, # samples in an extract, # frames in a sample, features) or (None, total frames, features).
        extract - typically 2s audio file that is loaded
        sample - typically ~10 ms audio that corresponds to a single input data
        """
        data_shape = self._determine_shape()
        total_desired_frames = data_shape[1] * data_shape[2]
        self._transformed = self._transformed[:, :total_desired_frames]
        if self._config["subtimesteps"]:
            self._transformed = self._transformed.reshape([data_shape[0], data_shape[1], data_shape[2], data_shape[3]])
        else:
            self._transformed = self._transformed.reshape([data_shape[0] * data_shape[1], data_shape[2], data_shape[3]])
