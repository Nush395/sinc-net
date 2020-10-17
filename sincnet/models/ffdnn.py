import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense

from .registry import register


@register("test_nn")
def simple_be(params):
    num_classes = 10

    inputs = Input((16,))
    x = Flatten()(inputs)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    y = Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, y)
    return model
