import tensorflow as tf
from unittest import TestCase
from sinc_net import SincConv1D


class TestSincConvLayer(TestCase):

    def test_initialisation_creates_odd_sized_filter(self):
        # given
        num_kernels = 4
        kernel_size = 4

        # when
        sinc_layer = SincConv1D(num_kernels, kernel_size)

        # then
        self.assertEqual(sinc_layer.kernel_size, 3)

    def test_layer_on_input(self):
        # given
        num_batches, num_samples = 3, 20
        sample_audio = tf.random.uniform((num_batches, 1, num_samples), seed=5)
        num_kernels, kernel_size = 4, 4
        sinc_layer = SincConv1D(num_kernels, kernel_size)

        # when
        sinc_layer(sample_audio)
