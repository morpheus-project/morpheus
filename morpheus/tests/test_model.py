# MIT License
# Copyright 2018 Ryan Hausen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ==============================================================================

"""Tests the model functions"""
import collections

import pytest
import numpy as np
import tensorflow as tf

import morpheus.core.base_model as base_model
import morpheus.core.unet as unet


class TestAssistant:
    """Makes things that the tests want."""

    @staticmethod
    def mock_dataset() -> collections.namedtuple:
        """Makes a compatible mock dataset.

        This snippet is adapted from morpheus.core.model.Morpheus
        """
        MockDataset = collections.namedtuple("Dataset", ["num_labels", "train", "test"])

        train_data = (TestAssistant.zero_array(), TestAssistant.zero_array())
        return MockDataset(5, train=train_data, test=train_data)

    @staticmethod
    def mock_hparams(
        inference: bool, batch_norm: bool, drop_out: bool, num_contractions: int
    ):
        """Makes a simple model config for testing.

        Args:
            inference (bool): boolean flag for inference
            batch_norm (bool): boolean flag for batch normalization in graph
            drop_out (bool): boolean flag for drop out in graph
            num_contractions (int): number of down/up sample pairs
        """

        num_filters = [1 for _ in range(num_contractions)]

        hparams = tf.contrib.training.HParams(
            inference=inference,
            num_epochs=1,
            batch_norm=batch_norm,
            drop_out=drop_out,
            down_filters=num_filters,
            up_filters=num_filters,
            learning_rate=0.1,
            dropout_rate=0.5,
            num_down_convs=1,
            num_up_convs=1,
            num_intermediate_filters=1,
        )

        return hparams

    @staticmethod
    def zero_tensor() -> tf.Tensor:
        """Makes a zero filled tensor with shape [3, 3]"""

        return tf.zeros([3, 3])

    @staticmethod
    def in_tensor(data_format: str) -> tf.Tensor:
        """Makes a sample tensor input tensor for shape testing.

        Args:
            data_format (str): 'channels_first' or 'channels_last'

        Returns:
            A tensor with shape [100, 80, 80, 5] if 'channels_last' or
            [100, 5, 80, 80] if 'channels_first'
        """

        shape = [100, 80, 80, 80]
        if data_format == "channels_first":
            shape[1] = 4
        else:
            shape[3] = 4

        return tf.zeros(shape, dtype=tf.float32)

    @staticmethod
    def zero_array() -> np.ndarray:
        """Makes a zero filled tensor with shape [3, 3]"""

        return np.zeros([3, 3])


@pytest.mark.unit
class TestBaseModel:
    """A class that tests the functions of morpheus.core.base_model.Model."""

    @staticmethod
    def test_model_fn_raises():
        """Tests that the non overridden model_fn raises NotImplemented."""

        model = base_model.Model(TestAssistant.mock_dataset())
        with pytest.raises(NotImplementedError):
            model.model_fn(TestAssistant.zero_tensor(), True)

    @staticmethod
    def test_build_graph_raises():
        """Tests that the non overridden model_fn raises on a build_graph call."""

        model = base_model.Model(TestAssistant.mock_dataset())
        with pytest.raises(NotImplementedError):
            model.build_graph(TestAssistant.zero_tensor(), True)

    @staticmethod
    def test_build_graph():
        """Tests the build_graph method."""

        model = base_model.Model(TestAssistant.mock_dataset())
        model.model_fn = lambda x, y: (x, y)
        expected_x = TestAssistant.zero_array()
        expected_is_training = True

        x, is_training = model.build_graph(expected_x, expected_is_training)

        np.testing.assert_array_equal(x, expected_x)
        assert is_training == expected_is_training

    @staticmethod
    def test_build_graph_singleton():
        """Tests the build_graph method, two calls."""

        model = base_model.Model(TestAssistant.mock_dataset())
        model.model_fn = lambda x, y: (x, y)
        expected_x = TestAssistant.zero_array()
        expected_is_training = True

        model.build_graph(expected_x, expected_is_training)
        x, is_training = model.build_graph(expected_x, expected_is_training)

        np.testing.assert_array_equal(x, expected_x)
        assert is_training == expected_is_training

    @staticmethod
    def test_train():
        """Test the train() method, doesn't raise."""
        model = base_model.Model(TestAssistant.mock_dataset())
        model.model_fn = lambda x, y: x
        model.train()

    @staticmethod
    def test_test():
        """Test the test() method, doesn't raise."""
        model = base_model.Model(TestAssistant.mock_dataset())
        model.model_fn = lambda x, y: x
        model.test()


@pytest.mark.unit
class TestUNet:
    """A class that tests the functions of morpheus.core.unet.Model"""

    @staticmethod
    def test_upsample_shape_doubles():
        inference, batch_norm, drop_out = True, True, True
        num_contractions = 1
        data_format = "channels_last"

        dataset = TestAssistant.mock_dataset()
        hparams = TestAssistant.mock_hparams(
            inference, batch_norm, drop_out, num_contractions
        )

        x = TestAssistant.in_tensor(data_format)

        expected_shape = x.shape.as_list()
        expected_shape[1] *= 2
        expected_shape[2] *= 2

        model = unet.Model(hparams, dataset, data_format)

        x = model.up_sample(x)

        actual_shape = x.shape.as_list()

        assert expected_shape == actual_shape

    @staticmethod
    def test_upsample_with_transpose_shape_doubles():
        inference, batch_norm, drop_out = True, True, True
        num_contractions = 1
        data_format = "channels_first"

        dataset = TestAssistant.mock_dataset()
        hparams = TestAssistant.mock_hparams(
            inference, batch_norm, drop_out, num_contractions
        )

        x = TestAssistant.in_tensor(data_format)

        expected_shape = x.shape.as_list()
        expected_shape[2] *= 2
        expected_shape[3] *= 2

        model = unet.Model(hparams, dataset, data_format)

        x = model.up_sample(x)

        actual_shape = x.shape.as_list()

        assert expected_shape == actual_shape

    @staticmethod
    def test_downsample_shape_halves():
        inference, batch_norm, drop_out = True, True, True
        num_contractions = 1
        data_format = "channels_last"

        dataset = TestAssistant.mock_dataset()
        hparams = TestAssistant.mock_hparams(
            inference, batch_norm, drop_out, num_contractions
        )

        x = TestAssistant.in_tensor(data_format)

        expected_shape = x.shape.as_list()
        expected_shape[1] /= 2
        expected_shape[2] /= 2

        model = unet.Model(hparams, dataset, data_format)

        x = model.down_sample(x)

        actual_shape = x.shape.as_list()

        assert expected_shape == actual_shape
