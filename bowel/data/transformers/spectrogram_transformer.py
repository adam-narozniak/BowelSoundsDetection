import librosa
import numpy as np
from scipy import signal
from tqdm import tqdm

from bowel.data.transformers.data_transformer import DataTransformer
from bowel.utils.audio_utils import split_spectrogram_to_windows


class SpectrogramTransformer(DataTransformer):
    """Creates spectrograms from audio data."""
    def __init__(self, data, config, expand_dims=False):
        super().__init__(data, config)
        self._expand_dims = expand_dims

    def _transform(self):
        spectrograms = []
        for audio in tqdm(self._data):
            s = self._create_spectrogram(audio)
            relative_frame_length = self._config["frame_length"] / self._config["sr"]
            s_windowed = np.array(split_spectrogram_to_windows(s, self._config["sample_length_seconds"], 1.0,
                                                               relative_frame_length))
            spectrograms.append(s_windowed)
        spectrograms = np.array(spectrograms)
        if self._expand_dims:
            # In case that the model starts with conv2d the data should be 4 dimensional (5 dims if time-distributed
            # is counted)
            self._transformed = np.expand_dims(spectrograms, -1)
        else:
            self._transformed = spectrograms

    def _create_spectrogram(self, sample):
        """Creates different kinds of spectrograms."""
        if self._config["mode"] == "magnitude" or self._config["mode"] == "psd":
            f, t, Sxx = signal.spectrogram(
                sample,
                window=self._config["window_type"],
                nperseg=self._config["frame_length"],
                noverlap=int(self._config["frame_length"] - self._config["hop_length"]),
                detrend="constant",
                scaling="density",
                mode=self._config["mode"]
            )
            f *= 44_100
            f_mask = (f <= self._config["max_freq"]) & (f >= self._config["min_freq"])
            return Sxx[f_mask]
        elif self._config["mode"] == "mel":
            return librosa.feature.melspectrogram(
                sample,
                sr=self._config["sr"],
                n_fft=self._config["frame_length"],
                hop_length=self._config["hop_length"],
                window=self._config["window_type"],
                center=True,
                fmax=self._config["max_freq"],
                n_mels=self._config["n_mels"]
            )
        else:
            raise ValueError(f"The given 'mode' in config is not supported. Given: {self._config['mode']}")
