from abc import abstractmethod, ABC

import numpy as np


class Normalizer(ABC):
    """Abstract class for normalization."""

    def __init__(self, normalize: bool = True):
        self._normalize: bool = normalize

    @abstractmethod
    def normalize(self, data: np.ndarray) -> None:
        """Normalize the data according to the strategy implemented in the subclass."""
        raise NotImplementedError()

    @abstractmethod
    def adapt(self, data: np.ndarray) -> np.ndarray:
        """Compute metrics on the train data that will be used for normalization."""
        raise NotImplementedError()
