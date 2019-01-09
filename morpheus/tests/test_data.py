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
"""Test the data functions"""
import os

from astropy.io import fits

import morpheus.data.example as example


class TestExample:
    """This tests the data.example module functions."""

    @staticmethod
    def test_sample_data_return():
        """Tests sample_data function returning arrays."""
        arrs = example.get_sample()

        expected_shape = (144, 144)
        for i in range(4):
            assert expected_shape == arrs[i].shape

    @staticmethod
    def test_sample_data_save():
        """Tests sample_data function saving to file."""
        local = os.path.dirname(os.path.abspath(__file__))

        example.get_sample(local)

        names = [f"{b}.fits" for b in ["h", "j", "v", "z"]]

        arrs = []
        for name in names:
            f_loc = os.path.join(local, name)
            arrs.append(fits.getdata(f_loc))
            os.remove(f_loc)

        expected_shape = (144, 144)
        for i in range(4):
            assert expected_shape == arrs[i].shape
