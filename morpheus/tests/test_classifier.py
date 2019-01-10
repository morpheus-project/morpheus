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
import os

import pytest
import numpy as np
import imageio
from astropy.io import fits

from morpheus.classifier import Classifier


@pytest.mark.unit
class TestClassifier:
    """Tests morpheus.classifier.Classifier"""

    @staticmethod
    def get_expected_pngs():
        expected_no_hidden_url = "https://drive.google.com/uc?export=download&id=1uGl6HOxVMa1L2-jU6kHhtz65hTX3-M1L"
        expected_hidden_url = "https://drive.google.com/uc?export=download&id=18CI48-ko8msRhkvVngvBWRvxL1fH3mE0"

        return {
            "no_hidden": imageio.imread(expected_no_hidden_url),
            "hidden": imageio.imread(expected_hidden_url),
        }

    @staticmethod
    def get_expected_output():
        expected_spheroid_url = "https://drive.google.com/uc?export=download&id=1nlGqibesE1LnEEif0oj-RO8xR4qNAFnP"
        expected_disk_url = "https://drive.google.com/uc?export=download&id=1btsoZZJu9qWkVn0rzIK9emgdMe6mMcHS"
        expected_irregular_url = "https://drive.google.com/uc?export=download&id=1qtXphVp7VflFBDWFjn6AJdvI1j3sN4KR"
        expected_point_source_url = "https://drive.google.com/uc?export=download&id=16bFNlZvD_EmAMSpCU-DZ_Shq3FMpXgp3"
        expected_background_url = "https://drive.google.com/uc?export=download&id=1xp6NC00T3JykdOwz0c8EFeJG0vdsThSW"
        expected_n_url = "https://drive.google.com/uc?export=download&id=1I5IasDPGyDmMaXN4NCwLh27X_OuxYiPv"

        return {
            "spheroid": fits.getdata(expected_spheroid_url),
            "disk": fits.getdata(expected_disk_url),
            "irregular": fits.getdata(expected_irregular_url),
            "point_source": fits.getdata(expected_point_source_url),
            "background": fits.getdata(expected_background_url),
            "n": fits.getdata(expected_n_url),
        }

    @staticmethod
    def test_variables_not_none():
        """Tests _variables_not_none method."""

        names = ["a", "b", "c"]
        values = [1, 2, 3]

        Classifier._variables_not_none(names, values)

    @staticmethod
    def test_variables_not_none_throws():
        """Tests _variables_not_none, throws ValueError."""

        names = ["a", "b", "c"]
        values = [1, 2, None]

        with pytest.raises(ValueError):
            Classifier._variables_not_none(names, values)

    @staticmethod
    def test_arrays_same_size():
        """Tests _arrays_same_size method."""

        shape = (10, 10)
        arrs = [np.zeros(shape) for _ in range(3)]

        Classifier._arrays_same_size(arrs)

    @staticmethod
    def test_arrays_same_size_throws():
        """Tests _arrays_same_size, throws ValueError."""

        shape = (10, 10)
        arrs = [np.zeros(shape) for _ in range(3)]
        arrs.append(np.zeros((20, 20)))

        with pytest.raises(ValueError):
            Classifier._arrays_same_size(arrs)

    @staticmethod
    def test_standardize_img():
        """Test _standardize_img method."""

        img = np.random.normal(loc=1.0, scale=3.0, size=(10, 10, 10))

        img = Classifier._standardize_img(img)

        np.testing.assert_allclose(np.mean(img), 0, atol=1e-07)
        np.testing.assert_allclose(np.var(img), 1, atol=1e-07)

    @staticmethod
    def test_make_runnable_file():
        """Test _make_runnable_file."""
        local = os.path.dirname(os.path.abspath(__file__))

        Classifier._make_runnable_file(local)

        assert os.path.exists(os.path.join(local, "main.py"))

        os.remove(os.path.join(local, "main.py"))

    @staticmethod
    def test_variables_not_none_raises():
        """Test _variables_not_none."""

        with pytest.raises(ValueError):
            Classifier._variables_not_none(["good", "bad"], [1, None])

    @staticmethod
    def test_colorize_rank_vote_output():
        """Test colorize_rank_vote_output."""

        data = TestClassifier.get_expected_output()
        expected_color = TestClassifier.get_expected_pngs()["no_hidden"]

        actual_color = Classifier.colorize_rank_vote_output(
            data, hide_unclassified=False
        )

        actual_color = (actual_color * 255).astype(np.uint8)

        np.testing.assert_array_almost_equal(expected_color, actual_color)

    @staticmethod
    def test_colorize_rank_vote_output_hidden():
        """Test colorize_rank_vote_output with hiding."""

        data = TestClassifier.get_expected_output()
        expected_color = TestClassifier.get_expected_pngs()["hidden"]

        actual_color = Classifier.colorize_rank_vote_output(
            data, hide_unclassified=True
        )

        actual_color = (actual_color * 255).astype(np.uint8)

        np.testing.assert_array_almost_equal(expected_color, actual_color)
