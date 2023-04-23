.PHONY: clean test data train crossval

all: clean data train test

data:
	rm -rf data/interim/*
	rm -rf data/processed/*
	python -m bowel.data.preprocess

clean:
	rm -rf data/interim/*
	rm -rf data/processed/*

train:
	python -m bowel.models.train train --model models/conv_rnn.h5

crossval:
	python -m bowel.models.train crossval --model models/conv_rnn.h5

test:
	python -m bowel.models.train test --model models/conv_rnn.h5 --config models/conv_rnn.yml
