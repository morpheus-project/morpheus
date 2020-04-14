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
import shutil
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
    def test_classify_image_mean_var():
        """Classify an image from files using mean and variance."""
        h, j, v, z = example.get_sample()

        outs = Classifier.classify(h=h, j=j, v=v, z=z, out_type="mean_var")

        expected_outs = dh.get_expected_morpheus_output(out_type="mean_var")

        for k in outs:
            np.testing.assert_allclose(
                outs[k], expected_outs[k], atol=1e-5, err_msg=f"{k} failed comparison"
            )

    @staticmethod
    def test_classify_image_mean_var_file():
        """Classify an image from files using mean and variance."""
        local = os.path.dirname(os.path.abspath(__file__))

        example.get_sample(local)

        h, j, v, z = [os.path.join(local, f"{b}.fits") for b in "hjvz"]

        Classifier.classify(h=h, j=j, v=v, z=z, out_dir=local, out_type="mean_var")

        outs = dh.get_expected_morpheus_output(out_type="mean_var")

        for k in outs:

            np.testing.assert_allclose(
                outs[k],
                fits.getdata(os.path.join(local, f"{k}.fits")),
                atol=1e-5,
                err_msg=f"{k} failed comparison",
            )

            os.remove(os.path.join(local, f"{k}.fits"))

        for b in "hjvz":
            os.remove(os.path.join(local, f"{b}.fits"))


@pytest.mark.parallel
class TestIntegrationParallel:
    @staticmethod
    def test_classify_rank_vote_parallel_cpu():
        """Classify an image in parallel with two cpus."""
        local = os.path.dirname(os.path.abspath(__file__))
        os.mkdir(os.path.join(local, "output"))
        out_dir = os.path.join(local, "output")

        example.get_sample(local)
        h, j, v, z = [os.path.join(local, f"{b}.fits") for b in "hjvz"]

        outs = dh.get_expected_morpheus_output(out_type="rank_vote")

        classified = Classifier.classify(
            h=h,
            j=j,
            v=v,
            z=z,
            out_dir=out_dir,
            out_type="rank_vote",
            cpus=2,
            parallel_check_interval=0.25,  # check every 15 seconds
        )

        for k in outs:
            np.testing.assert_allclose(
                outs[k], classified[k], atol=1e-5, err_msg=f"{k} failed comparison"
            )

        shutil.rmtree(out_dir)

        for b in [h, j, v, z]:
            os.remove(b)

    @staticmethod
    def test_classify_mean_var_parallel_cpu():
        """Classify an image in parallel with two cpus."""
        local = os.path.dirname(os.path.abspath(__file__))
        os.mkdir(os.path.join(local, "output"))
        out_dir = os.path.join(local, "output")

        example.get_sample(local)
        h, j, v, z = [os.path.join(local, f"{b}.fits") for b in "hjvz"]

        outs = dh.get_expected_morpheus_output(out_type="mean_var")

        classified = Classifier.classify(
            h=h,
            j=j,
            v=v,
            z=z,
            out_dir=out_dir,
            out_type="mean_var",
            cpus=2,
            parallel_check_interval=0.25,  # check every 15 seconds
        )

        for k in outs:
            np.testing.assert_allclose(
                outs[k], classified[k], atol=1e-5, err_msg=f"{k} failed comparison"
            )

        shutil.rmtree(out_dir)

        for b in [h, j, v, z]:
            os.remove(b)
