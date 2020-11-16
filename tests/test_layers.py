import tensorflow as tf
import numpy as np
import unittest
from statistics import mean, pstdev
from sincnet.layers.sinc import SincConv1D, LayerNorm


class TestSincConvLayer(unittest.TestCase):

    def test_initialisation_creates_odd_sized_filter(self):
        # given
        num_kernels = 4
        kernel_size = 4

        # when
        sinc_layer = SincConv1D(num_kernels, kernel_size)

        # then
        self.assertEqual(sinc_layer.kernel_size, 5)

    def test_layer_on_input(self):
        # given
        num_batches, num_samples = 3, 20
        sample_audio = tf.random.uniform((num_batches, 1, num_samples), seed=5)
        num_kernels, kernel_size = 4, 4
        sinc_layer = SincConv1D(num_kernels, kernel_size)

        # when
        sinc_layer(sample_audio)


class TestLayerNormLayer(unittest.TestCase):
    def test_layer_norm_builds_correct_trainable_params(self):
        # given
        ln = LayerNorm()
        input_shape = tf.TensorShape((5, 8))

        # when
        ln.build(input_shape)

        # then
        self.assertEqual(len(ln.trainable_weights), 2)
        self.assertEqual(ln.trainable_weights[0].shape.as_list(), [8])
        self.assertEqual(ln.trainable_weights[1].shape.as_list(), [8])

    def test_layer_call(self):
        # given
        data = [1, 2, 3, 4]
        inputs = tf.convert_to_tensor(data, dtype='float32')
        ln = LayerNorm()
        ln.build(inputs.shape)
        ln.epsilon = 0

        # when
        normed_inputs = ln.call(inputs)
        received_data = normed_inputs.numpy()

        # then
        mu, sg = mean(data), pstdev(data)
        sigma = pstdev(data)
        expected_data = np.array([(datum-mu)/sigma for datum in data])
        self.assertTrue(np.allclose(expected_data, received_data))
