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

from astropy.io import fits
import numpy as np
import pytest

from morpheus.classifier import Classifier
from morpheus.data import example


@pytest.mark.integration
class TestIntegration:
    """User level integration tests"""

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
    def test_classify_image():
        """User level example classification."""

        h, j, v, z = example.get_sample()

        expected_outs = TestIntegration.get_expected_output()

        outs = Classifier.classify_arrays(h=h, j=j, v=v, z=z, out_dir=None)

        for k in outs:
            np.testing.assert_allclose(
                outs[k], expected_outs[k], err_msg=f"{k} failed comparison"
            )
