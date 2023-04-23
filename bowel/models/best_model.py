import tensorflow as tf


class BestModel(tf.keras.Model):

    def __init__(self):
        super(BestModel, self).__init__()
        self._lstm_1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)))
        self._lstm_2 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=False)))
        self._drop_2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.7))
        self._conv_1 = tf.keras.layers.Conv1D(filters=128, kernel_size=15, activation='relu', padding='same', strides=1)
        self._classifier = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid'))

    def call(self, inputs, **kwargs):
        x = self._lstm_1(inputs)
        x = self._lstm_2(x)
        x = self._drop_2(x)
        x = self._conv_1(x)
        return self._classifier(x)
