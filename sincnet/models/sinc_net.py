import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, LayerNormalization,\
    Conv1D, BatchNormalization, LeakyReLU

from sincnet.data import AUDIO_SHAPE
from sincnet.layers.sinc import SincConv1D
from .registry import register


@register("sincnet_baseline")
def sincnet(hparams):
    """Baseline sincenet from M.Ravanelli paper."""
    inputs = Input(AUDIO_SHAPE)
    x = LayerNormalization()(inputs)
    x = SincConv1D(80, 251)(x)
    # TODO: LeakyReLU alpha not specified in paper, see what it is in code.
    x = LeakyReLU(alpha=0.01)(x)
    x = LayerNormalization()(x)
    x = Conv1D(60, 5)(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = LayerNormalization()(x)
    x = Conv1D(60, 5)(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Flatten()(x)

    def dense_block(input):
        x = Dense(2048)(input)
        x = LeakyReLU(alpha=0.01)(x)
        return BatchNormalization()(x)

    for i in range(3):
        x = dense_block(x)

    y = Dense(hparams.num_classes)(x)
    model = tf.keras.Model(inputs, y)
    return model
