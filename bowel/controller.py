import pathlib
import time
from typing import Optional, Any

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import wandb
from loguru import logger
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from wandb.keras import WandbMetricsLogger

from bowel.data.normalizers.all_normalizer import AllNormalizer
from bowel.factories import instantiate_transformer, instantiate_normalizers, create_experimental_model
from bowel.models.best_model import BestModel
from bowel.models.lstm_model import LSTM_model
from bowel.models.lstm_with_conv_model import LSTM_with_conv_model
from bowel.utils.io import save_yaml
from bowel.utils.metrics import f1
from bowel.utils.metrics import get_mean_scores
from bowel.utils.wandb_utils import set_custom_wandb_summary


class Controller:
    """Trains, test or cross-validates models."""

    def __init__(self,
                 transform_config: dict[str: Any],
                 train_config: dict[str: Any],
                 data: pd.DataFrame,
                 labels: np.ndarray
                 ):
        # configs
        self.transform_config = transform_config
        self.train_config = train_config
        self.full_config = {**train_config, **transform_config}

        # data
        self._kfold_list = list(range(1, self.train_config["kfold"] + 1))
        self._features_type = self.transform_config["features_type"]
        self._data = data
        self._labels = labels
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # model
        self.model_type = self.train_config["model_type"]
        self.model = None

        self.data_transformer = None
        self.audio_normalizer = None
        self.features_normalizer = None

        # eval
        self._train_part = None
        self.train_history = None
        self.train_evaluation_score = None
        self.test_evaluation_score = None

        # booking
        self.time_book = pd.DataFrame(columns=["start_time", "end_time"])  # index will be the procedure type

    def instantiate_model(self, model_type: str, train_config: dict[Any:Any]) -> None:
        """Creates model based on the model_type and others parameters specified in the train configuration file."""
        if model_type == "experimental":
            model = create_experimental_model(train_config)
        elif model_type == "best_mfcc_lstm":
            model = BestModel()
        elif model_type == "mfcc_lstm":
            model = LSTM_model()
        elif model_type == "spec_lstm_with_conv":
            model = LSTM_with_conv_model()
        elif model_type == "dummy":
            model = DummyClassifier(strategy="most_frequent")
        elif model_type == "mean_std_reg":
            model = LogisticRegression(solver="saga",
                                       max_iter=train_config["epochs"],
                                       class_weight=train_config["class_weight"])
        else:
            raise ValueError(f"The given 'model_type' is not supported. Given {model_type}.")

        if isinstance(model, tf.keras.Model):
            model.compile(loss=train_config["loss"], optimizer=train_config["optimizer"],
                          metrics=[
                              "accuracy",
                              f1,
                              tf.keras.metrics.Precision(name="precision"),
                              tf.keras.metrics.Recall(name="recall"),
                              tf.keras.metrics.AUC(curve='PR', name="auc"),
                              tf.keras.metrics.TruePositives(name="TP"),
                              tf.keras.metrics.TrueNegatives(name="TN"),
                              tf.keras.metrics.FalsePositives(name="FP"),
                              tf.keras.metrics.FalseNegatives(name="FN")
                          ])
        self.model = model

    def _create_train_test_mask(self, test_fold: int):
        """Based on the folds (given in the data) creates masks ready to used for indexing."""
        kfold_list = self._kfold_list.copy()
        kfold_list.remove(test_fold)
        train_kfold_list = kfold_list
        train_mask = np.isin(self._data.index.get_level_values('kfold').values, train_kfold_list)
        test_mask = self._data.index.get_level_values('kfold').values == test_fold
        return train_mask, test_mask

    def create_train_test_data(self,
                               test_fold: Optional[int] = None):
        """Given loaded data (optionally) normalizes audio and transforms into the features (optionally normalized)."""
        if test_fold is None:
            test_fold = self._kfold_list[-1]
        train_mask, test_mask = self._create_train_test_mask(test_fold)
        train_part = int(train_mask.shape[0] * (1. - self.train_config["validation_split"]))
        self._train_part = train_part
        data = np.array([row[0] for row in self._data.values])
        self.audio_normalizer, self.features_normalizer = instantiate_normalizers(self.transform_config)
        self.audio_normalizer.adapt(data[train_mask][:train_part])
        data[train_mask] = self.audio_normalizer.normalize(data[train_mask])
        data[test_mask] = self.audio_normalizer.normalize(data[test_mask])
        # move this to external function
        self.data_transformer = instantiate_transformer(data, self.transform_config, self.model_type)
        transformed_data = self.data_transformer.transformed
        self.features_normalizer.adapt(transformed_data[train_mask][:train_part])
        transformed_data[train_mask] = self.features_normalizer.normalize(transformed_data[train_mask])
        transformed_data[test_mask] = self.features_normalizer.normalize((transformed_data[test_mask]))
        if self.transform_config["subtimesteps"]:
            pass
        else:  # this thing needs to be checked
            # previously ..., self._label_creator.n_subsamples)
            train_mask = np.repeat(train_mask, transformed_data.shape[0] // train_mask.shape[0])
            test_mask = np.repeat(test_mask.reshape(-1, 1), transformed_data.shape[0] // test_mask.shape[0])
        self.X_train, self.X_test = transformed_data[train_mask], transformed_data[test_mask]
        self.y_train, self.y_test = self._labels[train_mask], self._labels[test_mask]

    def train_test(self):
        """Firstly trains the data on the first n-1 folds, then tests it on the remaining one fold."""
        st = time.time()
        self.train_evaluation_score = self.train()
        et = time.time()
        self.time_book.loc[0, "train_time"] = st - et
        st = time.time()
        self.test_evaluation_score = self.test()
        et = time.time()
        self.time_book.loc[0, "test_time"] = st - et

    def crossval(self, group_id):
        """
        Perform n-fold cross-validation.

        Returns:
            mean test scores
        """
        train_scores = []
        test_scores = []
        total_folds = len(self._kfold_list[:-1])
        current_fold = 1
        for test_fold in self._kfold_list[:-1]:  # the last "fold" is the test set
            wandb.init(group=group_id, config=self.full_config)
            logger.info(f"Cross-validation on fold {current_fold}/{total_folds} started")
            self.create_train_test_data(test_fold)
            train_score = self.train()
            train_scores.append(train_score)
            test_score = self.test()
            test_scores.append(test_score)
            if test_fold != self._kfold_list[-2]:
                # Don't close wandb to save average results in the last run
                wandb.finish()
            logger.info(f"Cross-validation on fold {current_fold}/{total_folds} done")
            current_fold += 1
        self.train_evaluation_score = get_mean_scores(train_scores)
        logger.info("Average train score on the cross-validation:")
        logger.info(self.train_evaluation_score)
        self.test_evaluation_score = get_mean_scores(test_scores)
        logger.info("Average test score on the cross-validation:")
        logger.info(self.test_evaluation_score)
        return self.test_evaluation_score

    def train(self):
        """
        Trains the model based on the params specified in configs.

        Returns:
            train evaluation score: the score calculated on the same data it was trained on
        """
        logger.info("Training started")
        self.instantiate_model(self.model_type, self.train_config)
        if isinstance(self.model, tf.keras.Model):
            self.train_history = self.model.fit(self.X_train,
                                                self.y_train,
                                                validation_split=self.train_config["validation_split"],
                                                epochs=self.train_config["epochs"],
                                                callbacks=[
                                                    WandbMetricsLogger(),
                                                    tf.keras.callbacks.EarlyStopping(
                                                        monitor="val_loss",
                                                        patience=self.train_config["patience"],
                                                        restore_best_weights=True
                                                    )])
        elif isinstance(self.model, sklearn.linear_model.LogisticRegression) \
                or isinstance(self.model, sklearn.dummy.DummyClassifier):
            self.model.fit(self.X_train,
                           self.y_train)
        else:
            raise TypeError(f"The argument type doesn't meet the expected ones, it is {type(self.model)} ")
        logger.info("Training done")
        self.train_evaluation_score = self.model.evaluate(self.X_train[:self.train_part],
                                                          self.y_train[:self.train_part], return_dict=True)
        logger.info(f"Evaluation score on train data:")
        logger.info(f"{self.train_evaluation_score}")
        set_custom_wandb_summary(self.train_evaluation_score, "train/")
        return self.train_evaluation_score

    def test(self):
        """
        Test only already initialized/loaded model.

        Returns:
            test evaluation scores: the score calculated on the test data (on which model was not trained)
        """
        logger.info("Testing started")
        test_evaluation_score = self.model.evaluate(self.X_test, self.y_test, return_dict=True)
        logger.info("Testing done")
        logger.info(f"Evaluation score on test data:")
        logger.info(f"{test_evaluation_score}")
        set_custom_wandb_summary(test_evaluation_score, prefix="test/")
        return test_evaluation_score

    def test_from_saved(self, test_fold: int = None, save_path: pathlib.Path = None):
        if test_fold is None:
            test_fold = self._kfold_list[-1]
        train_mask, test_mask = self._create_train_test_mask(test_fold)
        data = np.array([row[0] for row in self._data.values])
        X_test = self.audio_normalizer.normalize(data[test_mask]).numpy()
        self.data_transformer = instantiate_transformer(X_test, self.transform_config, self.model_type)
        transformed_data = self.data_transformer.transformed
        transformed_data = self.features_normalizer.normalize(transformed_data)
        self.X_test = transformed_data
        self.y_test = self._labels[test_mask]
        self.model = tf.keras.models.load_model(save_path, custom_objects={"f1": f1})
        self.test()

    def save_model(self, path: pathlib.Path):
        """Saves only keras model in SavedModel format."""
        if isinstance(self.model, tf.keras.Model):
            tf.keras.models.save_model(self.model, path)
            self._save_norm_params(path / "norm.yaml")
        elif isinstance(self.model, sklearn.linear_model.LogisticRegression) \
                or isinstance(self.model, sklearn.dummy.DummyClassifier):
            pass
        else:
            raise TypeError(f"The argument type doesn't meet the expected ones, it is {type(self.model)} ")

    def _save_norm_params(self, path: pathlib.Path):
        """Mean and variance used in the normalization need to be saved for saved model use (models from a disc)."""
        try:
            normalizer_params = {
                "audio_mean": float(self.audio_normalizer.normalizer.mean.numpy()[0]),
                "audio_var": float(self.audio_normalizer.normalizer.variance.numpy()[0])
            }
        except AttributeError:  # when the Normalizer has not called adapt, no mean attribute is available
            normalizer_params = {
                "audio_mean": 0.0,
                "audio_var": 1.0
            }

        if isinstance(self.features_normalizer, AllNormalizer):
            normalizer_params["mean_mfcc"] = float(self.features_normalizer.normalizer.mean.numpy()[0])
            normalizer_params["var_mfcc"] = float(self.features_normalizer.normalizer.variance.numpy()[0])
        save_yaml(normalizer_params, path)
