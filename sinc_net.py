import tensorflow as tf
import math
import numpy as np


class SincConv1D(tf.keras.layers.Layer):

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, num_kernels, kernel_size, sample_rate=16000, stride=1,
                 padding='VALID', dilation=1, min_low_hz=50, min_band_hz=50):
        """ Initialise the SincConv1D layer.

        Args:
            num_kernels: Number of filters
            kernel_size: Filter length.
            sample_rate: Sample rate. Defaults to 16000.
            stride:
            padding:
            dilation:
            min_low_hz:
            min_band_hz:
        """
        super().__init__()
        self.num_kernels = num_kernels
        # force the filters to be odd (to be symmetric)
        self.kernel_size = kernel_size + 1 if not kernel_size % 2 else kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.num_kernels + 1)
        hz = self.to_hz(mel)

        # filter lower frequency
        self.low_hz_ = tf.Variable(hz[:-1], trainable=True)
        # filter band frequency
        self.band_hz_ = tf.Variable(np.diff(hz), trainable=True)

        # Hamming Window - only need to compute half of window
        n_lin = tf.linspace(0.0, float((self.kernel_size / 2) - 1),
                            int((self.kernel_size / 2)))
        self.window_ = 0.54 - 0.46 * tf.math.cos(2 * math.pi * n_lin
                                                 / self.kernel_size)

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        # Due to symmetry, I only need half of the time axes
        self.n_ = 2 * math.pi * tf.range(-n, 0) / self.sample_rate

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """Do a forward pass of the SincConv1D layer.

        Args:
            inputs: (batch_size, 1, n_samples) Batch of waveforms.
        Returns:
        """
        low = self.min_low_hz + tf.math.abs(self.low_hz_)
        high = tf.clip_by_value(low +
                                self.min_band_hz +
                                tf.math.abs(self.band_hz_),
                                self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = tf.linalg.matmul(low, self.n_)
        f_times_t_high = tf.linalg.matmul(high, self.n_)

        band_pass_left = ((tf.math.sin(f_times_t_high) -
                           tf.math.sin(f_times_t_low))
                          / (self.n_ / 2)) * self.window_
        band_pass_center = 2 * band
        band_pass_right = tf.reverse(band_pass_left, [1])

        band_pass = tf.concat([band_pass_left,
                               band_pass_center,
                               band_pass_right],
                              1)

        band_pass = band_pass / (2 * band[:, None])
        filters = band_pass.reshape(self.kernel_size,1, self.out_channels)

        return tf.nn.conv1d(inputs, filters, stride=self.stride,
                            padding=self.padding, dilations=self.dilation,
                            data_format='NCW')

    def get_config(self):
        super().get_config()
