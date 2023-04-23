from hyperopt import hp

experiments_params = {
    # architecture
    "normalize_audio": True,
    "block1": {"if_block": True, "units": hp.quniform("units1", 64, 256, q=64),
               "bidirectional": hp.choice("bidirectional1", [False, True]),
               "dropout": hp.choice("dropout1",
                                    [
                                        {"if_dropout": False},
                                        {"if_dropout": True, "rate": hp.quniform("dropout_rate1", 20, 80, 5) / 100.}
                                    ])},

    "block2": hp.choice("if_block2",
                        [
                            {"if_block": False, "block3": {"if_block": False}},
                            {"if_block": True, "units": hp.quniform("units2", 64, 256, q=64),
                             "bidirectional": hp.choice("bidirectional2", [False, True]),
                             "dropout": hp.choice("dropout2",
                                                  [
                                                      {"if_dropout": False},
                                                      {"if_dropout": True,
                                                       "rate": hp.quniform("dropout_rate2", 20, 80, 5) / 100.}
                                                  ]),
                             "block3": hp.choice("if_block3",
                                                 [
                                                     {"if_block": False},
                                                     {"if_block": True, "units": hp.quniform("units3", 64, 256, q=64),
                                                      "bidirectional": hp.choice("bidirectional3", [False, True]),
                                                      "dropout": hp.choice("dropout3",
                                                                           [
                                                                               {"if_dropout": False},
                                                                               {"if_dropout": True,
                                                                                "rate": hp.quniform("dropout_rate3", 20,
                                                                                                    80, 5) / 100.}
                                                                           ])}

                                                 ])
                             }

                        ]),

    "convolution_at_end": hp.choice("convolution_at_end",
                                    [
                                        {"convolution_at_end": False},
                                        {"convolution_at_end": True,
                                         "filters": hp.quniform("end_conv_units1", low=32, high=256, q=32),
                                         "kernel_size": hp.quniform("end_conv_kernel_size1", low=9, high=27, q=3)}

                                    ]),

    # transformation
    "window_type": hp.choice("window_type", ["hamming", "hann", "triang", "bartlett", "tukey", "lanczos"]),
    "max_freq": hp.quniform("max_freq", low=1_000, high=10_000, q=500),
    "n_mfcc": hp.quniform("n_mfcc", low=10, high=128, q=2),
    "frame_length": hp.choice("frame_length", [440, 880]),
    "hop_length": hp.choice("hop_length", [55, 110, 220, 440]),

    # static from train config
    "epochs": 200,
    "validation_split": 0.15,
    "patience": 10,
    "model_type": "experimental",
    "loss": "binary_crossentropy",
    "optimizer": "adam",

    # static from transformation config
    "features_type": "mfcc",
    "normalize_features": True,
    "sr": 44100,
    "sample_length_seconds": 2,
    "kfold": 5,
    "subtimesteps": True

}
