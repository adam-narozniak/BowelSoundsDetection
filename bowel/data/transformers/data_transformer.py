from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from loguru import logger


class DataTransformer(ABC):
    """Abstract class that transforms audio data into a features that the model will be fed with."""

    def __init__(self, data: pd.DataFrame, config: dict):
        """
        Args:
            data: pd.DataFrame index with "kfold" and "filename" index pointing to single channel audio-file
            config: transformation config with parameters needed to apply it (specific for each subclass)
                e.g. fft, hop_length

        """
        self._data = data
        self._config = config
        self._transformed: Union[np.ndarray, None] = None

    @abstractmethod
    def _transform(self):
        raise NotImplementedError

    @property
    def transformed(self) -> Union[np.ndarray, None]:
        """Returns the transformed data if available otherwise transforms it first and then returns."""
        if self._transformed is None:
            logger.info("Data transformation started")
            self._transform()
            logger.info("Data transformation done")
            return self._transformed
        else:
            return self._transformed
