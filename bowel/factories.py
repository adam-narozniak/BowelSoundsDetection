import tensorflow as tf

from bowel.data.normalizers.all_normalizer import AllNormalizer
from bowel.data.transformers.audio_features_transformer import AudioFeaturesTransformer
from bowel.data.transformers.mean_std_transformer import MeanStdTransformer
from bowel.data.transformers.raw_audio_transformer import RawAudioTransformer
from bowel.data.transformers.spectrogram_transformer import SpectrogramTransformer
from bowel.utils.io import load_config


def instantiate_transformer(data, transform_config, model_type):
    """Creates object to change audio files into features for a neural network."""
    feat_type = transform_config["features_type"]
    if feat_type == "mfcc":
        data_transformer = AudioFeaturesTransformer(data, transform_config)
    elif feat_type == "spec" and model_type != "spec_lstm_with_conv":
        data_transformer = SpectrogramTransformer(data, transform_config)
    elif feat_type == "spec":
        data_transformer = SpectrogramTransformer(data, transform_config, expand_dims=True)
    elif feat_type == "raw":
        data_transformer = RawAudioTransformer(data, transform_config)
    elif feat_type == "mean_std":
        data_transformer = MeanStdTransformer(data, transform_config)
    else:
        raise ValueError(f"The given 'features_type' in config is not supported. Given: {feat_type}")
    return data_transformer


def instantiate_normalizers(transform_config, mode: str = "train", save_path=None):
    """Creates normalizers to be adapted or uses previously computed mean and variance (test mode)"""
    if mode == "test":
        config = load_config(save_path / "norm.yaml")
        audio_normalizer = AllNormalizer(normalize=transform_config["normalize_audio"],
                                         mean=config["audio_mean"], variance=config["audio_var"])
        if transform_config["features_type"] == "mfcc":
            features_normalizer = AllNormalizer(normalize=transform_config["normalize_features"],
                                                mean=config["mean_mfcc"],
                                                variance=config["var_mfcc"])
        else:
            features_normalizer = AllNormalizer(normalize=transform_config["normalize_features"],
                                                mean=config["feature_mean"],
                                                variance=config["feature_variance"])
    else:
        audio_normalizer = AllNormalizer(normalize=transform_config["normalize_audio"])
        features_normalizer = AllNormalizer(normalize=transform_config["normalize_features"])
    return audio_normalizer, features_normalizer


def create_experimental_model(params):
    # create the model
    model = tf.keras.Sequential()

    if_blocks = [params["block1"]["if_block"], params["block2"]["if_block"], params["block2"]["block3"]["if_block"]]
    ret_seq: list[bool]
    if sum(if_blocks) == 1:
        ret_seq = [False]
    elif sum(if_blocks) == 2:
        ret_seq = [True, False]
    else:
        ret_seq = [True, True, False]
    lstm1 = tf.keras.layers.LSTM(units=int(params["block1"]["units"]), return_sequences=ret_seq[0])
    if params["block1"]["bidirectional"]:
        lstm1 = tf.keras.layers.Bidirectional(lstm1)
    model.add(tf.keras.layers.TimeDistributed(lstm1))

    if params["block1"]["dropout"]["if_dropout"]:
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(rate=params["block1"]["dropout"]["rate"])))

    if params["block2"]["if_block"]:
        lstm2 = tf.keras.layers.LSTM(units=int(params["block2"]["units"]), return_sequences=ret_seq[1])
        if params["block2"]["bidirectional"]:
            lstm2 = tf.keras.layers.Bidirectional(lstm2)
        model.add(tf.keras.layers.TimeDistributed(lstm2))
        if params["block2"]["dropout"]["if_dropout"]:
            model.add(
                tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(rate=params["block2"]["dropout"]["rate"])))

        if params["block2"]["block3"]["if_block"]:
            lstm3 = tf.keras.layers.LSTM(units=int(params["block2"]["block3"]["units"]), return_sequences=ret_seq[2])
            if params["block2"]["block3"]["bidirectional"]:
                lstm3 = tf.keras.layers.Bidirectional(lstm3)
            model.add(tf.keras.layers.TimeDistributed(lstm3))
            if params["block2"]["block3"]["dropout"]["if_dropout"]:
                model.add(tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dropout(rate=params["block2"]["block3"]["dropout"]["rate"])))

    if params["convolution_at_end"]["convolution_at_end"]:
        model.add(tf.keras.layers.Conv1D(filters=int(params["convolution_at_end"]["filters"]),
                                         kernel_size=int(params["convolution_at_end"]["kernel_size"]),
                                         activation='relu',
                                         padding='same', strides=1))

    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid')))

    return model
