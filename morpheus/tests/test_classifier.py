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

from morpheus.classifier import Classifier
from morpheus.data import example

import morpheus.tests.data_helper as dh


@pytest.mark.unit
class TestClassifier:
    """Tests morpheus.classifier.Classifier"""

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

    # New API ==================================================================
    @staticmethod
    def test_validate_parallel_params_raises_cpus_gpus():
        """Test _validate_parallel_params.

        Throws ValueError for passing values for both cpus an gpus.
        """
        gpus = [0]
        cpus = 0

        with pytest.raises(ValueError):
            Classifier._validate_parallel_params(gpus=gpus, cpus=cpus)

    @staticmethod
    def test_validate_parallel_params_raises_single_gpu():
        """Test _validate_parallel_params.

        Throws ValueError for passing a single gpu.
        """
        gpus = [0]

        with pytest.raises(ValueError):
            Classifier._validate_parallel_params(gpus=gpus)

    @staticmethod
    def test_validate_parallel_params_raises_single_cpu():
        """Test _validate_parallel_params.

        Throws ValueError for passing a single gpu.
        """
        cpus = 1

        with pytest.raises(ValueError):
            Classifier._validate_parallel_params(cpus=cpus)

    @staticmethod
    def test_segmap_from_classified():
        """Test the segmap_from_classified method."""

        data = dh.get_expected_morpheus_output()
        h, _, _, _ = example.get_sample()
        mask = np.zeros_like(h, dtype=np.int)
        mask[5:-5, 5:-5] = 1

        expected_segmap = dh.get_expected_segmap()["segmap"]

        actual_segmap = Classifier.segmap_from_classifed(data, h, mask=mask)

        np.testing.assert_array_equal(expected_segmap, actual_segmap)

    @staticmethod
    def test_catalog_from_classified():
        """Test the catalog_from_classified method."""

        classified = dh.get_expected_morpheus_output()
        h, _, _, _ = example.get_sample()
        segmap = dh.get_expected_segmap()["segmap"]

        expected_catalog = dh.get_expected_catalog()["catalog"]

        actual_catalog = Classifier.catalog_from_classified(classified, h, segmap)

        assert expected_catalog == actual_catalog

    @staticmethod
    def test_colorize_classified():
        """Test colorize_classified."""

        data = dh.get_expected_morpheus_output()
        expected_color = dh.get_expected_colorized_pngs()["no_hidden"]

        actual_color = Classifier.colorize_classified(data, hide_unclassified=False)

        actual_color = (actual_color * 255).astype(np.uint8)

        np.testing.assert_array_almost_equal(expected_color, actual_color)

    @staticmethod
    def test_colorize_classified_hidden():
        """Test colorize_classified with hidden."""

        classified = dh.get_expected_morpheus_output()
        expected_color = dh.get_expected_colorized_pngs()["hidden"]

        actual_color = Classifier.colorize_classified(
            classified, hide_unclassified=True
        )

        actual_color = (actual_color * 255).astype(np.uint8)

        np.testing.assert_array_almost_equal(expected_color, actual_color)

    @staticmethod
    def test_valid_input_types_is_str_ndarray():
        """Test _valid_input_types_is_str."""

        h, j, v, z = [np.zeros([10]) for _ in range(4)]

        assert not Classifier._valid_input_types_is_str(h, j, v, z)

    @staticmethod
    def test_valid_input_types_is_str_str():
        """Test _valid_input_types_is_str."""

        h, j, v, z = ["" for _ in range(4)]

        assert Classifier._valid_input_types_is_str(h, j, v, z)

    @staticmethod
    def test_valid_input_types_is_str_throws_mixed():
        """Test _valid_input_types_is_str."""

        h, j = ["" for _ in range(2)]
        v, z = [np.zeros([10]) for _ in range(2)]

        with pytest.raises(ValueError):
            Classifier._valid_input_types_is_str(h, j, v, z)

    @staticmethod
    def test_valid_input_types_is_str_throws_wrong_type():
        """Test _valid_input_types_is_str."""

        h, j, v, z = [1 for _ in range(4)]

        with pytest.raises(ValueError):
            Classifier._valid_input_types_is_str(h, j, v, z)

    # New API ==================================================================
