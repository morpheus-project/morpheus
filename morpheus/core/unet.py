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
"""Implements variations of the U-Net architecture."""

import tensorflow.compat.v1 as tf

import morpheus.core.base_model
from morpheus.core.hparams import HParams

LAYERS = tf.layers
VarInit = tf.variance_scaling_initializer


class Model(morpheus.core.base_model.Model):
    """Based on U-Net (https://arxiv.org/abs/1505.04597).

    Args:
        hparams (tf.contrib.training.HParams): Hyperparamters to use
        dataset (tf.data.Dataset): dataset to use for training
        data_format (str): channels_first or channels_last

    Required HParams:
        down_filters (list): number of filters for each down conv section
        num_down_convs (int): number of conv ops per down conv section
        up_filters (list): number of filters for each up conv section
        num_up_convs (int): number of conv ops per up conv section
        batch_norm (bool): use batch normalization
        dropout (bool): use dropout

    Optional HParams:
        dropout_rate (float): the percentage of neurons to drop [0.0, 1.0]
    """

    def __init__(
        self,
        hparams: HParams,
        dataset: tf.data.Dataset,
        data_format="channels_last",
    ):
        """Inits Model with hparams, dataset, and data_format"""

        super().__init__(dataset, data_format)
        self.hparams = hparams

    def model_fn(self, inputs: tf.Tensor, is_training: bool) -> tf.Tensor:
        """Defines U-Net graph using HParams.

        Args:
            inputs (tf.Tensor): The input tensor to the graph
            is_training (bool): indicates if the model is in the training phase

        Returns:
            tf.Tensor: the output tensor from the graph

        TODO: add input shape check for incompatible tensor shapes
        """

        outputs = []

        for idx, num_filters in enumerate(self.hparams.down_filters):
            with tf.variable_scope("downconv-{}".format(idx), reuse=tf.AUTO_REUSE):
                for c_idx in range(self.hparams.num_down_convs):
                    with tf.variable_scope(
                        "conv-{}".format(c_idx), reuse=tf.AUTO_REUSE
                    ):
                        inputs = self.block_op(inputs, num_filters, is_training)

                outputs.append(inputs)
                inputs = self.down_sample(inputs)

        with tf.variable_scope("intermediate-conv", reuse=tf.AUTO_REUSE):
            inputs = self.block_op(
                inputs, self.hparams.num_intermediate_filters, is_training
            )

        concat_axis = 3 if self.data_format == "channels_last" else 1
        for idx, num_filters in enumerate(self.hparams.up_filters):
            with tf.variable_scope("upconv-{}".format(idx), reuse=tf.AUTO_REUSE):
                inputs = self.up_sample(inputs)
                inputs = tf.concat(
                    [inputs, outputs[-(idx + 1)]], concat_axis, name="concat_op"
                )
                for c_idx in range(self.hparams.num_up_convs):
                    with tf.variable_scope(
                        "conv-{}".format(c_idx), reuse=tf.AUTO_REUSE
                    ):
                        inputs = self.block_op(inputs, num_filters, is_training)

        with tf.variable_scope("final_conv", reuse=tf.AUTO_REUSE):
            inputs = self.conv(
                inputs, self.dataset.num_labels, activation=None, kernel_size=3
            )

        return inputs

    def block_op(
        self, inputs: tf.Tensor, num_filters: int, is_training: bool
    ) -> tf.Tensor:
        """Basic unit of work batch_norm->conv->dropout.

        Batch normalization and dropout are conditioned on the obect's HParams

        Args:
            inputs (tf.Tensor): input tensor
            num_filters (int): number of inputs for the conv operation
            is_training: indicates if the model is training

        Returns:
            tf.Tensor: the output tensor from the block operation
        """

        if self.hparams.batch_norm:
            inputs = self.batch_norm(inputs, is_training)

        inputs = self.conv(inputs, num_filters)

        if self.hparams.dropout:
            inputs = self.dropout(inputs, is_training)

        return inputs

    def batch_norm(
        self, inputs: tf.Tensor, is_training: bool
    ):  # pylint: disable=missing-docstring
        axis = 3 if self.data_format == "channels_last" else 1

        return LAYERS.batch_normalization(inputs, training=is_training, axis=axis)

    def dropout(
        self, inputs: tf.Tensor, is_training: bool
    ):  # pylint: disable=missing-docstring
        rate = self.hparams.dropout_rate if is_training else 0
        return LAYERS.dropout(inputs, rate=rate)

    def conv(
        self,
        inputs,
        num_filters,
        padding="same",
        strides=1,
        activation=tf.nn.relu,
        name="conv",
        kernel_size=3,
    ):  # pylint: disable=missing-docstring,too-many-arguments

        inputs = LAYERS.conv2d(
            inputs,
            num_filters,
            kernel_size,
            padding=padding,
            strides=strides,
            kernel_initializer=VarInit,
            use_bias=True,
            data_format=self.data_format,
            name=name,
            activation=activation,
        )

        return inputs

    def down_sample(self, inputs):
        """Reduces inputs width and height by half.

        Args:
            inputs (tf.Tensor): input tensor

        Returns:
            input tensor downsampled
        """
        pool_size = 2
        stride = 2
        return LAYERS.max_pooling2d(
            inputs, pool_size, stride, data_format=self.data_format, name="downsampler"
        )

    def up_sample(self, inputs):
        """Doubles inputs width and height.

        Transposes the input if necessary for tf.image.resize_images

        Args:
            inputs (tf.Tensor): input tensor

        Returns:
            input tensor upsampled
        """

        def wrap_tranpose(up_func, _inputs):
            _inputs = tf.transpose(_inputs, [0, 2, 3, 1])
            _inputs = up_func(_inputs)
            _inputs = tf.transpose(_inputs, [0, 3, 1, 2])

            return _inputs

        def upsample_func(_inputs):
            _, width, height, _ = _inputs.shape.as_list()

            _inputs = tf.image.resize_images(
                _inputs,
                (width * 2, height * 2),
                method=tf.image.ResizeMethod.BICUBIC,
                align_corners=True,
            )

            return _inputs

        if self.data_format == "channels_first":
            return wrap_tranpose(upsample_func, inputs)

        return upsample_func(inputs)
