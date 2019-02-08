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
"""Integration tests for Morpheus"""

import os

import numpy as np
import pytest
from astropy.io import fits

import morpheus.tests.data_helper as dh
from morpheus.classifier import Classifier
from morpheus.data import example


@pytest.mark.integration
class TestIntegration:
    """User level integration tests."""

    @staticmethod
    def test_classify_array():
        """User level example classification of in memory array."""

        h, j, v, z = example.get_sample()

        expected_outs = dh.get_expected_morpheus_output()

        outs = Classifier.classify_arrays(h=h, j=j, v=v, z=z, out_dir=None)

        for k in outs:
            np.testing.assert_allclose(
                outs[k], expected_outs[k], err_msg=f"{k} failed comparison"
            )

    @staticmethod
    def test_classify_file():
        """User level example classification of in memmapped array."""

        out_dir = "tmp"
        if out_dir not in os.listdir():
            os.mkdir(out_dir)

        example.get_sample(out_dir=out_dir)
        h, j, v, z = [os.path.join(out_dir, f"{b}.fits") for b in "hjvz"]

        Classifier.classify_files(h=h, j=j, v=v, z=z, out_dir=out_dir)

        expected_outs = dh.get_expected_morpheus_output()

        for k in expected_outs:
            actual = fits.getdata(os.path.join(out_dir, f"{k}.fits"))
            np.testing.assert_allclose(
                actual, expected_outs[k], err_msg=f"{k} failed comparison"
            )

        # clean up
        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, f))

        os.rmdir(out_dir)

    @staticmethod
    def test_make_catalog():
        """User level catalog."""

        h, j, v, z = example.get_sample()

        catalog = Classifier.catalog_arrays(h=h, j=j, z=z, v=v)

        expected_catalog = dh.get_expected_catalog()

        def element_equal(val, exp_val):
            if isinstance(val, int):
                assert val == exp_val
            if isinstance(val, float):
                np.testing.assert_almost_equal(val, exp_val)
            if isinstance(val, list):
                assert len(val) == len(exp_val)

                for v, e in zip(val, exp_val):
                    element_equal(v, e)

        assert catalog.keys() == expected_catalog.keys()

        for k in catalog:
            element_equal(catalog[k], expected_catalog[k])
