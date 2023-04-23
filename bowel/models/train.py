import os
import argparse

import yaml
import numpy as np
import wandb

from bowel.models.conv_rnn import ConvRnn
from bowel.models.conv_merge import ConvMerge
from bowel.data.load import Loader
from bowel.utils.train_utils import get_score, get_scores_mean
from bowel.config import *


class Trainer:
    """A class to train and test models.
    """
    def __init__(self, data_dir, model_file, config):
        """Trainer constructor.

        Args:
            data_dir (str): Path to directory with processed data.
            model_file (str): Path to model file.
            config (dict): Dictionary with config parameters.
        """
        self.config = config
        self.data_dir = data_dir
        self.parts = list(range(1, self.config['kfold'] + 1))
        self.model_file = model_file
        if config['model_type'] == 'convrnn':
            self.Model = ConvRnn
        if config['model_type'] == 'convmerge':
            self.Model = ConvMerge

    def train(self):
        """Train model, print metrics on trained model and save model to file.
        """
        train_loader = Loader(self.data_dir, self.config, self.parts[:-1])
        test_loader = Loader(self.data_dir, self.config, [self.parts[-1]])
        X_train, y_train = train_loader.get_data()
        X_test, y_test = test_loader.get_data()
        model = self.Model(X_train[0].shape, self.config)
        model.train(X_train, y_train, X_test, y_test)
        model.save(self.model_file)
        print(model.summary())
        y_train_pred = model.predict(X_train)
        print('train data:')
        print(get_score(y_train, y_train_pred))
        y_test_pred = model.predict(X_test)
        print('test data:')
        print(get_score(y_test, y_test_pred))

    def crossval(self):
        """Do kfold crossvalidation, print averaged metrics and save last produced model to file.
        """
        train_scores = []
        test_scores = []
        for i in range(self.config['kfold']):
            train_loader = Loader(self.data_dir, self.config,
                                  self.parts[:i] + self.parts[i + 1:])
            test_loader = Loader(self.data_dir, self.config, [self.parts[i]])
            X_train, y_train = train_loader.get_data()
            X_test, y_test = test_loader.get_data()
            model = self.Model(X_train[0].shape, self.config)
            model.build_model()
            model.train(X_train, y_train, X_test, y_test)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_scores.append(get_score(y_train, y_train_pred))
            test_scores.append(get_score(y_test, y_test_pred))
            print('train data:')
            print(train_scores[-1])
            print('test data:')
            print(test_scores[-1])
        model.save(self.model_file)
        print(model.summary())
        print('train data:')
        print(get_scores_mean(train_scores))
        print('test data:')
        print(get_scores_mean(test_scores))

    def test(self):
        """Print metrics on loaded trained model.
        """
        train_loader = Loader(self.data_dir, self.config, self.parts[:-1])
        test_loader = Loader(self.data_dir, self.config, [self.parts[-1]])
        X_train, y_train = train_loader.get_data()
        X_test, y_test = test_loader.get_data()
        model = self.Model(X_train[0].shape, self.config, self.model_file)
        print(model.summary())
        y_train_pred = model.predict(X_train)
        print('train data:')
        print(get_score(y_train, y_train_pred))
        y_test_pred = model.predict(X_test)
        print('test data:')
        print(get_score(y_test, y_test_pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str,
                        help='"train": training model, "crossval": k-fold crossvalidation, "test": testing model on dataset')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='path to data directory')
    parser.add_argument('--model', type=str, default='models/model.h5',
                        help='if "train" and "crossval" mode: path to save model, if "test": path to load model')
    parser.add_argument('--config', type=str, default='bowel/config.yml',
                        help='yaml file with data and model parameters')
    parser.add_argument('--dryrun', action='store_true',
                        help='not sync run with wandb')
    parser.add_argument('--experiment', type=str, default=None,
                        help='name of wandb experiment')
    args = parser.parse_args()
    np.random.seed(10)
    config = yaml.safe_load(open(args.config))
    if args.dryrun:
        os.environ['WANDB_MODE'] = 'dryrun'
    wandb.init(project=WANDB_PROJECT_NAME, config=config)
    if args.experiment is not None:
        wandb.run.name = args.experiment
    trainer = Trainer(args.data_dir, args.model, config)
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'crossval':
        trainer.crossval()
    elif args.mode == 'test':
        trainer.test()
