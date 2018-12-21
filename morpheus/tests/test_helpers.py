"""
MIT License
Copyright 2018 Ryan Hausen

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""Tests the helper functions"""

import pytest
import tensorflow as tf

from morpheus.core.helpers import TFLogger
from morpheus.core.helpers import OptionalFunc


class TestTFLogger:
    """This tests the TFLogger class' funtions

    TODO: Figure out to properly test this.
    """

    def test_info(self):
        """Tests TFlogger.info"""
        TFLogger.info("Test Message")

    def test_debug(self):
        """Tests TFlogger.debug"""
        TFLogger.debug("Test Message")

    def test_warn(self):
        """Tests TFlogger.warn"""
        TFLogger.warn("Test Message")

    def test_error(self):
        """Tests TFlogger.error"""
        TFLogger.error("Test Message")

    def test_tensor_shape(self):
        """Tests TFlogger.tensor_shape"""
        t = tf.zeros([3, 3], dtype=tf.int32, name="test")

        TFLogger.tensor_shape(t)
