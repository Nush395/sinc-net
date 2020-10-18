import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, LayerNormalization

from sincnet.layers.logmelspec import LogMelSpectrogram
from .registry import register


@register("test_nn")
def simple_be(params):
    num_classes = 12

    inputs = Input((16000,))
    x = LogMelSpectrogram()(inputs)
    x = LayerNormalization(axis=2, name='batch_norm')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    y = Dense(num_classes)(x)
    model = tf.keras.Model(inputs, y)
    return model
