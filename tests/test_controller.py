import pathlib
import unittest

from hamcrest import *

from bowel.controller import Controller
from bowel.data.data_loader import DataLoader
from bowel.data.label_creator import LabelCreator
from bowel.models.best_model import BestModel
from bowel.utils.io import load_divsion_file, load_config


class TestController(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Creates controller"""

        CONFIG_DIR = pathlib.Path("./../configs/")
        DATA_DIR = pathlib.Path("./../data/processed/")
        # path to the file specifying samples names and division in folds
        DIVISION_FILE_PATH = pathlib.Path("./../data/processed/files.csv")
        transform_config_name = "best_model_transformation.yaml"
        train_config_name = "best_model_train.yaml"

        # data loading
        transform_config_path = CONFIG_DIR / transform_config_name
        train_config_path = CONFIG_DIR / train_config_name
        self.transform_config = load_config(transform_config_path)
        self.train_config = load_config(train_config_path)
        full_config = {**self.train_config, **self.transform_config}
        division_data = load_divsion_file(DIVISION_FILE_PATH).iloc[:3]

        # useful classes initializations
        data_loader = DataLoader(DATA_DIR, division_data)
        label_creator = LabelCreator(DATA_DIR, division_data, self.transform_config)
        data = data_loader.load_data()
        labels = label_creator.labels

        self.controller = Controller(
            self.transform_config,
            self.train_config,
            data,
            labels
        )

    def test_instantiate_model_best(self):
        self.controller.instantiate_model("best_mfcc_lstm", self.train_config)
        assert_that(self.controller.model, instance_of(BestModel))

    def test_instantiate_model_raises(self):
        with self.assertRaises(ValueError) as context:
            self.controller.instantiate_model("non-existent-model", self.train_config)
        self.assertTrue("The given 'model_type' is not supported. Given non-existent-model" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
