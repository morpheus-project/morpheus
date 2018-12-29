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
"""Tests morpheus.classifier module."""

import pytest

from morpheus.classifier import Classifier

class TestClassifier:
    """Tests morpheus.classifier.Classifier"""

    @staticmethod
    def test_variables_not_none():
        """Tests Classifier._variables_not_none."""

        names = ['a', 'b', 'c']
        values = [1, 2, 3]

        Classifier._variables_not_none(names, values)

    @staticmethod
    def test_variables_not_none_throws():
        """Tests Classifier._variables_not_none, throws ValueError."""

        names = ['a', 'b', 'c']
        values = [1, 2, None]

        with pytest.raises(ValueError):
            Classifier._variables_not_none(names, values)