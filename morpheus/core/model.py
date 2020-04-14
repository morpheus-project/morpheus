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
"""Contains model code for Morpheus."""

import collections
import json
import os
from typing import List

import tensorflow.compat.v1 as tf

import morpheus.core.unet
from morpheus.core.hparams import HParams


class Morpheus(morpheus.core.unet.Model):
    """The main class for the Morpheus model.

    This class takes a HParams object as an argument and it should
    contain the following properties:

    Note if you are using pretrained weights for inference only you need
    to mock the dataset object and use the default hparams.

    You can mock the dataset object calling Morpheus.mock_dataset().

    You can get the default HParams by calling Morpheus.inference_hparams().

    An example call for inference only

    >>> dataset = Morpheus.mock_dataset()
    >>> hparams = Morpheus.inference_hparams()
    >>> data_format = 'channels_last'
    >>> morph = Morpheus(hparams, dataset, data_format)

    Required HParams:
        * inference (bool): true if using pretrained model
        * down_filters (list): number of filters for each down conv section
        * num_down_convs (int): number of conv ops per down conv section
        * up_filters (list): number of filters for each up conv section
        * num_up_convs (int): number of conv ops per up conv section
        * batch_norm (bool): use batch normalization
        * dropout (bool): use dropout

    Optional HParams:
        * learning_rate (float): learning rate for training, required if inference is set to false
        * dropout_rate (float): the percentage of neurons to drop [0.0, 1.0]

    Args:
        hparams (morpheus.core.hparams.HParams): Model Hyperparameters
        dataset (tf.data.Dataset): dataset to use for training
        data_format: channels_first or channels_last

    TODO:
        * Make optimizer a parameter
    """

    def __init__(
        self, hparams: HParams, dataset: tf.data.Dataset, data_format: str,
    ):
        """Inits Morpheus with hparams, dataset, data_format."""

        super().__init__(hparams, dataset, data_format)
        if not hparams.inference:

            self.opt = tf.train.AdamOptimizer(hparams.learning_rate)

    def loss_func(self, logits: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        """Defines the loss function used in training.

        The loss function is defined by combining cross entropy loss calculated
        against all 5 classes and dice loss calculated against just the
        background class.

        Args:
            logits (tf.Tensor): output tensor from graph should be
                                [batch_size, width, height, 5]
            labels (tf.Tensor): labels used in training should be
                                [batch_size, width, height, 5]

        Returns:
            tf.Tensor: Tensor representing loss function.
        """

        flat_logits = tf.reshape(logits, [-1, 5])
        flat_y = tf.reshape(labels, [-1, 5])

        # Calculate weighted crossentropy ======================================
        # This is normally calculated by taking a count of the pixels assigned
        # to each class, but because we have continous values for each class
        # we sum the probabilities for each class in the pixels instead.
        xentropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=flat_logits, labels=flat_y
        )

        dominant_class = tf.argmax(flat_y, axis=1, output_type=tf.int32)
        p_dominant_class = tf.reduce_max(flat_y, axis=1)

        class_coefficient = tf.zeros_like(xentropy_loss)
        for output_class_idx in range(5):
            class_pixels = tf.cast(
                tf.equal(output_class_idx, dominant_class), tf.float32
            )

            coef = tf.reduce_mean(class_pixels * p_dominant_class)
            class_coefficient = tf.add(class_coefficient, coef * class_pixels)

        class_coefficient = 1 / class_coefficient

        weighted_xentropy_loss = tf.reduce_mean(xentropy_loss * class_coefficient)
        # Calculate weighted crossentropy ======================================

        # Calculate dice loss ==================================================
        if self.data_format == "channels_first":
            yh_background = tf.nn.sigmoid(logits[:, -1, :, :])
            y_background = labels[:, -1, :, :]
        else:
            yh_background = tf.nn.sigmoid(logits[:, :, :, -1])
            y_background = labels[:, :, :, -1]

        dice_numerator = tf.reduce_sum(y_background * yh_background, axis=[1, 2])
        dice_denominator = tf.reduce_sum(y_background + yh_background, axis=[1, 2])

        dice_loss = tf.reduce_mean(2 * dice_numerator / dice_denominator)
        # Calculate dice loss ==================================================

        total_loss = weighted_xentropy_loss
        total_loss = total_loss + (1 - dice_loss)

        return total_loss

    def optimizer(self, loss: tf.Tensor) -> tf.Tensor:
        """Overrides the optimizer func in morpheus.core.unet

        Args:
            loss (tf.Tensor): The loss function tensor to pass to the optimizer

        Returns:
            tf.Tensor: the Tensor result of optimizer.minimize()
        """
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimize = self.opt.minimize(loss)

        return optimize

    def train_metrics(
        self, logits: tf.Tensor, labels: tf.Tensor
    ) -> (
        (List[str], List[tf.Tensor]),
        List[tf.Tensor],
    ):  # overrides base method pylint: disable=no-self-use
        """Overrides the train_metrics func in morpheus.core.unet

        Args:
            logits (tf.Tensor): the output logits from the model
            labels (tf.Tensor): the labels used during training

        Returns:
            Tuple(Tuple(
                    List(str): names of metrics,
                    List(tf.Tensor): tensors for metrics
                  ),
                  List(tf.Tensor): Tensors for updating  running metrics
        """
        with tf.name_scope("train_metrics"):
            metrics_dict = Morpheus.eval_metrics(logits, labels)

        names, finalize, running = [], [], []

        for key in sorted(metrics_dict):
            names.append(key)
            finalize.append(metrics_dict[key][0])
            running.append(metrics_dict[key][1])

        return ([names, finalize], running)

    def test_metrics(
        self, logits: tf.Tensor, labels: tf.Tensor
    ) -> (
        (List[str], List[tf.Tensor]),
        List[tf.Tensor],
    ):  # overrides base method pylint: disable=no-self-use
        """Overrides the test_metrics func in morpheus.core.unet

        Args:
            logits (tf.Tensor): the output logits from the model
            labels (tf.Tensor): the labels used during training

        Returns:
            Tuple(Tuple(
                    List(str): names of metrics,
                    List(tf.Tensor): tensors for metrics
                  ),
                  List(tf.Tensor): Tensors for updating  running metrics
        """
        with tf.name_scope("test_metrics"):
            metrics_dict = Morpheus.eval_metrics(logits, labels)

        names, finalize, running = [], [], []

        for key in sorted(metrics_dict):
            names.append(key)
            finalize.append(metrics_dict[key][0])
            running.append(metrics_dict[key][1])

        return ([names, finalize], running)

    def inference(self, inputs: tf.Tensor) -> tf.Tensor:
        """Performs inference on input.

        Args:
            inputs (tf.Tensor): input tensor with shape
                                [batch_size, width, height, 5]

        Returns:
            A tf.Tensor of [batch_size, width, height, 5] representing the
            output the model, includes applying the softmax function.
        """

        return tf.nn.softmax(self.build_graph(inputs, False))

    @staticmethod
    def eval_metrics(yh: tf.Tensor, y: tf.Tensor) -> dict:
        """Function to generate metrics for evaluation during training.

        Args:
            yh (tf.Tensor): network output [n,h,w,c]
            y (tf.Tensor):  labels         [n,h,w,c]

        Returns:
            A dictionary collection of (tf.Tensor, tf.Tensor), where the keys
            are the names of the metrics and the values are running metric
            pairs. More infor on running accuracy metrics here:
            https://www.tensorflow.org/api_docs/python/tf/metrics/accuracy
        """
        metrics_dict = {}

        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        classes = ["spheroid", "disk", "irregular", "point_source", "background"]

        yh_bkg = tf.reshape(tf.nn.sigmoid(yh[:, :, :, -1]), [-1])
        y_bkg = tf.reshape(y[:, :, :, -1], [-1])
        for threshold in thresholds:
            name = "iou-{}".format(threshold)
            with tf.name_scope(name):
                preds = tf.cast(tf.greater_equal(yh_bkg, threshold), tf.int32)
                metric, update_op = tf.metrics.mean_iou(y_bkg, preds, 2, name=name)
                metrics_dict[name] = (metric, update_op)

        # Calculate the accuracy per class per pixel
        y = tf.reshape(y, [-1, 5])
        yh = tf.reshape(yh, [-1, 5])
        lbls = tf.argmax(y, 1)
        preds = tf.argmax(yh, 1)

        name = "overall"
        metric, update_op = tf.metrics.accuracy(lbls, preds, name=name)

        metrics_dict[name] = (metric, update_op)
        for i, _ in enumerate(classes):
            in_c = tf.equal(lbls, i)
            name = classes[i]
            metric, update_op = tf.metrics.accuracy(
                lbls, preds, weights=in_c, name=name
            )
            metrics_dict[name] = (metric, update_op)

        return metrics_dict

    @staticmethod
    def mock_dataset() -> collections.namedtuple:
        """Generates a mockdataset for inference.

        Returns:
            A collections.namedtuple object that can be passed in place of a
            tf.data.Dataset for 'dataset' argument in the constructor
        """
        MockDataset = collections.namedtuple("Dataset", ["num_labels"])
        return MockDataset(5)

    @staticmethod
    def inference_hparams() -> HParams:
        """Generates a mockdataset for inference.

        Returns:
            a morpheus.core.hparams.HParams object with the settings for inference
        """
        config_path = os.path.join(os.path.dirname(__file__), "model_config.json")

        with open(config_path, "r") as f:
            return HParams(**json.load(f))

    @staticmethod
    def get_weights_dir() -> str:
        """Returns the location of the weights for tf.Saver."""
        return os.path.join(os.path.dirname(__file__), "model_weights")
