import numpy as np
import tensorflow as tf

from bowel.data.normalizers.normalizer import Normalizer


class AllNormalizer(Normalizer):
    """
    Normalizes features by computing single mean and variance for the data.
    """

    def __init__(self, normalize: bool = True, mean: float = None, variance: float = None) -> None:
        super().__init__(normalize)
        self.normalizer = tf.keras.layers.Normalization(axis=None, mean=mean, variance=variance)

    def adapt(self, data: np.ndarray) -> None:
        """Adapt single global mean and variance."""
        if self._normalize is False:
            pass
        else:
            self.normalizer.adapt(data)

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize to global mean and variance."""
        if self._normalize is False:
            return data
        else:
            return self.normalizer(data)
