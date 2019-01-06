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
"""A base class for building neural network models in Tensorflow."""

from types import FunctionType

import tensorflow as tf

from morpheus.core.helpers import OptionalFunc


class Model:
    """ Base class for models.

    Attributes:
        dataset (tf.data.Dataset): Dataset Object for training
        is_training (bool): indicates whether or not the model is training
        data_format (str): 'channels_first' or 'channels_last'

    Required methods to override:
        model_fn: the graph function

    Optional methods to override:
        train_metrics: to add metrics during training
        test_metrics: to add metrics during testing, can be same as train_metrics
        optimizer: updates params based on a loss tensor
        loss_func: defines a loss value given and x and y tensor
        inference: default applies softmax to tensor from model_fn
    """

    train_metrics = OptionalFunc("No training metrics set")
    test_metrics = OptionalFunc("No test metrics set")
    optimizer = OptionalFunc("No optimizer set")
    loss_func = OptionalFunc("No loss function set")
    inference = OptionalFunc("Nor inference fucntion set")

    def __init__(self, dataset: tf.data.Dataset, data_format: str = "channels_last"):
        """Inits Model with dataset, and data_format"""
        self.dataset = dataset
        self.data_format = data_format
        self._graph = None

    def model_fn(self, inputs: tf.Tensor, is_training: bool) -> FunctionType:
        """Function that defines model. Needs to be Overidden!

        Args:
            inputs (tf.Tensor): the input tensor
            is_training (bool): boolean to indicate if in training phase

        Returns:
            Should return a function that takes two inputs tf.Tensor and bool

        Raises:
            NotImplementedError if not overridden
        """
        raise NotImplementedError()

    def build_graph(self, inputs: tf.Tensor, is_training: bool) -> tf.Tensor:
        """Base function that returns model_fn evaluated on x. Don't Override!

        Args:
            inputs (tf.Tensor): The tensor to be processed, ie a placeholder
            is_training (bool): whether or not the model is training useful
                                for things like batch normalization or
                                dropout

        Returns:
            returns the tensor that represents the result of model_fn evaluated
            on the input tensor

        Raises
            NotImplementedError if Model.model_fn() is not overwritten
        """
        if self._graph:
            return self._graph(inputs, is_training)

        self._graph = self.model_fn
        return self._graph(inputs, is_training)

    def train(self) -> (tf.Tensor, tf.Tensor):
        """Builds the training routine tensors. Don't Override!

        Returns:
            (optimize, metrics): the result of self.optimizer and
                                 self.train_metrics respectively

        Raises
            NotImplementedError if Model.model_fn() is not overwritten
        """
        data, labels = self.dataset.train
        logits = self.build_graph(data, True)
        optimize = self.optimizer(self.loss_func(logits, labels))
        metrics = self.train_metrics(logits, labels)

        return optimize, metrics

    def test(self) -> (tf.Tensor, tf.Tensor):
        """Builds the testing routing tensors. Don't Override!

        Returns:
            (logits, metrics): the result of the self.build_graph and
                               self.test_metrics respectively

        Raises
            NotImplementedError if Model.model_fn() is not overwritten
        """
        inputs, labels = self.dataset.test
        logits = self.build_graph(inputs, False)
        metrics = self.test_metrics(logits, labels)

        return logits, metrics
