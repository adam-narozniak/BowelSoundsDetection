"""Utility models to make sure that the GPUs can be accessed by TensorFlow."""
import tensorflow as tf
from loguru import logger


def log_tensorflow_gpu_availability() -> None:
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        logger.debug(f"There are NO GPUs available for Tensorflow.")
    else:
        logger.debug(f"GPUs for Tensorflow detected: {gpus}")
