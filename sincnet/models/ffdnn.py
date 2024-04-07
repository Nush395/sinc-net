import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, LayerNormalization

from sincnet.layers.logmelspec import LogMelSpectrogram
from sincnet.data import AUDIO_SHAPE
from .registry import register


@register("test_nn")
def simple_be(hparams):

    inputs = Input(AUDIO_SHAPE)
    x = LogMelSpectrogram()(inputs)
    x = LayerNormalization(axis=2, name='batch_norm')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    y = Dense(hparams.num_classes)(x)
    model = tf.keras.Model(inputs, y)
    return model
