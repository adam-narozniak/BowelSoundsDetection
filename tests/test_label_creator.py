import pathlib
import unittest

from hamcrest import *
from parameterized import parameterized

from bowel.data.label_creator import LabelCreator
from bowel.utils.io import load_divsion_file


class LabelCreatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        CONFIG_DIR = pathlib.Path("./../configs/")
        DATA_DIR = pathlib.Path("./../data/processed/")
        transform_config_name = "best_model_transformation.yaml"
        # path to the file specifying samples names and division in folds
        DIVISION_FILE_PATH = pathlib.Path("./../data/processed/files.csv")

        # data loading
        transform_config = {"frame_length": 50, "sr": 100, "sample_length_seconds": 2, "hop_length": 1}
        division_data = load_divsion_file(DIVISION_FILE_PATH)

        self.label_creator = LabelCreator(DATA_DIR, division_data, transform_config)

    @parameterized.expand([
        ([1., 1.5, 0., 0.5], 0),  # the search period ends before the start of the sound
        ([0.4, 0.9, 1.0, 1.5], 0),  # the search period starts after the sound end
        ([1.01, 1.27, 1.0, 1.5], 1),
        # search period is bigger than the sound (sound is fully present is search period) and sound is longer than the half of hte frame
        ([1.01, 1.11, 1., 1.5], 0),
        # search period is bigger than the sound (sound is fully present is search period) and the soound is shorter than a half of the frame
        ([1.01, 1.6, 1., 1.5], 1),
        # search period ends before the end of the sound or search period starts after the sound starts
    ])
    def test_if_sound_present(self, x, y):
        result = self.label_creator._if_sound_present(*x)
        assert_that(result, is_(y))


if __name__ == '__main__':
    unittest.main()
