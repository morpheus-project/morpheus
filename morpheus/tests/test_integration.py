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
from morpheus.classifier import Classifier
from morpheus.data import example

import morpheus.tests.data_helper as dh


@pytest.mark.integration
class TestIntegration:
    """User level integration tests."""

    @staticmethod
    def test_classify_image_rank_vote_in_mem():
        """Classify an image in memory using rank vote."""

        h, j, v, z = example.get_sample()

        expected_outs = dh.get_expected_morpheus_output()

        outs = Classifier.classify(h=h, j=j, v=v, z=z, out_dir=None)

        for k in outs:
            np.testing.assert_allclose(
                outs[k], expected_outs[k], err_msg=f"{k} failed comparison"
            )

    @staticmethod
    def test_classify_image_rank_vote_file():
        """Classify an image from files using rank vote."""
        local = os.path.dirname(os.path.abspath(__file__))

        example.get_sample(local)

        h, j, v, z = [os.path.join(local, f"{b}.fits") for b in "hjvz"]

        Classifier.classify(h=h, j=j, v=v, z=z, out_dir=local)

        outs = dh.get_expected_morpheus_output()

        for k in outs:

            np.testing.assert_allclose(
                outs[k],
                fits.getdata(os.path.join(local, f"{k}.fits")),
                err_msg=f"{k} failed comparison",
            )

            os.remove(os.path.join(local, f"{k}.fits"))

        for b in "hjvz":
            os.remove(os.path.join(local, f"{b}.fits"))

    @staticmethod
    def test_classify_image_mean_var_file():
        """Classify an image from files using mean and variance."""
        local = os.path.dirname(os.path.abspath(__file__))

        example.get_sample(local)

        h, j, v, z = [os.path.join(local, f"{b}.fits") for b in "hjvz"]

        # Classifier.classify(h=h, j=j, v=v, z=z, out_dir=local, out_type="mean_var")

        # outs = dh.get_expected_morpheus_output()

        # for k in outs:

        #     np.testing.assert_allclose(
        #         outs[k],
        #         fits.getdata(os.path.join(local, f"{k}.fits")),
        #         err_msg=f"{k} failed comparison"
        #     )

        #     os.remove(os.path.join(local, f"{k}.fits"))

        # for b in "hjvz":
        #     os.remove(os.path.join(local, f"{b}.fits"))
