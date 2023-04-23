import pathlib
import unittest

import numpy as np
from hamcrest import *

from bowel.data.data_loader import DataLoader
from bowel.utils.io import load_divsion_file


class TestDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Creates elements needed for data loader"""
        self.data_dir = pathlib.Path("./../data/processed/")
        self.division_data = load_divsion_file(pathlib.Path("./../data/processed/files.csv")).iloc[:3]
        self.data_loader = DataLoader(self.data_dir, self.division_data)

    def test_load_single_file_first_ret(self):
        single_file_data = self.data_loader._load_single_file(0)
        assert_that(single_file_data[0][0], instance_of(np.int64))

    def test_load_single_file_second_ret(self):
        single_file_data = self.data_loader._load_single_file(0)
        assert_that(single_file_data[0][1], instance_of(str))

    def test_load_single_file_third_ret(self):
        single_file_data = self.data_loader._load_single_file(0)
        assert_that(single_file_data[1][0], instance_of(np.ndarray))

    def test_load_data(self):
        data = self.data_loader.load_data()
        datum = data.iloc[0].values
        assert_that(datum.shape, is_((1,)))

    def test_load_data_sample_length(self):
        data = self.data_loader.load_data()
        datum = data.iloc[0].values
        assert_that(datum[0].shape, is_((44_100,)))


if __name__ == '__main__':
    unittest.main()
