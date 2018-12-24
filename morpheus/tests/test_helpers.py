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

"""Tests the helper functions"""

import os
import pytest
import numpy as np
import tensorflow as tf
from astropy.io import fits

from morpheus.core.helpers import FitsHelper
from morpheus.core.helpers import TFLogger
from morpheus.core.helpers import OptionalFunc


class TestTFLogger:
    """This tests the TFLogger class' funtions.

    TODO: Figure out to properly test this.
    """

    @staticmethod
    def test_info():
        """Tests TFlogger.info"""
        TFLogger.info("Test Message")

    @staticmethod
    def test_debug():
        """Tests TFlogger.debug"""
        TFLogger.debug("Test Message")

    @staticmethod
    def test_warn():
        """Tests TFlogger.warn"""
        TFLogger.warn("Test Message")

    @staticmethod
    def test_error():
        """Tests TFlogger.error"""
        TFLogger.error("Test Message")

    @staticmethod
    def test_tensor_shape():
        """Tests TFlogger.tensor_shape"""
        t = tf.zeros([3, 3], dtype=tf.int32, name="test")

        TFLogger.tensor_shape(t)


class TestOptionalFunc:
    """This tests OptionalFunc class."""

    @staticmethod
    def test_placeholder():
        """Tests the dummy function in OptionalFunc"""
        args = [1, 2, 3]

        assert args == OptionalFunc.placeholder(args)

    @staticmethod
    def test_default_is_set():
        """Tests that the default flag is set on init for OptionalFunc"""

        of = OptionalFunc("Test")
        of.__get__(None)
        assert of._is_default

    @staticmethod
    def test_default_is_set_off():
        """Tests that the default flag is set on init for OptionalFunc"""

        of = OptionalFunc("Test")
        of.__set__(None, lambda x: x)

        assert not of._is_default


class TestFitsHelper:
    """This tests the FITS helper class."""

    @staticmethod
    def test_create_file_rasies():
        """Tests that a ValueError is raised from improper dtype param"""

        with pytest.raises(ValueError):
            FitsHelper.create_file("dummy.fits", [0], str)

    @staticmethod
    def test_create_unit8_file():
        """Tests 2d uint8 file creation"""
        local = os.path.dirname(os.path.abspath(__file__))
        dummy_name = os.path.join(local, "dummy_unit8.fits")
        dummy_shape = (200, 200)
        dummy_dtype = np.uint8

        FitsHelper.create_file(dummy_name, dummy_shape, dummy_dtype)

        retrieved = fits.getdata(dummy_name)

        os.remove(dummy_name)

        assert dummy_shape == retrieved.shape
        assert np.issubdtype(dummy_dtype, retrieved.dtype)

    @staticmethod
    def test_create_int16_file():
        """Tests 2d int16 file creation"""
        local = os.path.dirname(os.path.abspath(__file__))
        dummy_name = os.path.join(local, "dummy_int16.fits")
        dummy_shape = (200, 200)
        dummy_dtype = np.int16

        FitsHelper.create_file(dummy_name, dummy_shape, dummy_dtype)

        retrieved = fits.getdata(dummy_name)

        os.remove(dummy_name)

        assert dummy_shape == retrieved.shape
        assert np.issubdtype(dummy_dtype, retrieved.dtype)

    @staticmethod
    def test_create_int32_file():
        """Tests 2d int32 file creation"""
        local = os.path.dirname(os.path.abspath(__file__))
        dummy_name = os.path.join(local, "dummy_int32.fits")
        dummy_shape = (200, 200)
        dummy_dtype = np.int32

        FitsHelper.create_file(dummy_name, dummy_shape, dummy_dtype)

        retrieved = fits.getdata(dummy_name)

        os.remove(dummy_name)

        assert dummy_shape == retrieved.shape
        assert np.issubdtype(dummy_dtype, retrieved.dtype)

    @staticmethod
    def test_create_float32_file():
        """Tests 2d float32 file creation"""
        local = os.path.dirname(os.path.abspath(__file__))
        dummy_name = os.path.join(local, "dummy_float32.fits")
        dummy_shape = (200, 200)
        dummy_dtype = np.float32

        FitsHelper.create_file(dummy_name, dummy_shape, dummy_dtype)

        retrieved = fits.getdata(dummy_name)

        os.remove(dummy_name)

        assert dummy_shape == retrieved.shape
        assert np.issubdtype(dummy_dtype, retrieved.dtype)

    @staticmethod
    def test_create_float64_file():
        """Tests 2d float64 file creation"""
        local = os.path.dirname(os.path.abspath(__file__))
        dummy_name = os.path.join(local, "dummy_float64.fits")
        dummy_shape = (200, 200)
        dummy_dtype = np.float64

        FitsHelper.create_file(dummy_name, dummy_shape, dummy_dtype)

        retrieved = fits.getdata(dummy_name)

        os.remove(dummy_name)

        assert dummy_shape == retrieved.shape
        assert np.issubdtype(dummy_dtype, retrieved.dtype)

    @staticmethod
    def test_create_file_3d():
        """Tests a 3d file creation"""
        local = os.path.dirname(os.path.abspath(__file__))
        dummy_name = os.path.join(local, "dummy_unit8_3d.fits")
        dummy_shape = (200, 200, 5)
        dummy_dtype = np.uint8

        FitsHelper.create_file(dummy_name, dummy_shape, dummy_dtype)

        retrieved = fits.getdata(dummy_name)

        os.remove(dummy_name)

        # the channels when stored as fits get moved to the front.
        dummy_shape = tuple(reversed(dummy_shape))
        assert dummy_shape == retrieved.shape
        assert np.issubdtype(dummy_dtype, retrieved.dtype)

    @staticmethod
    def test_get_files():
        """Tests the get_files() method"""
        local = os.path.dirname(os.path.abspath(__file__))

        dummy_file1 = os.path.join(local, "dummy1.fits")
        dummy_file2 = os.path.join(local, "dummy2.fits")

        arr1 = np.zeros([100, 100], dtype=np.float32)
        arr2 = np.ones([100, 100], dtype=np.float32)

        for f, a in zip([dummy_file1, dummy_file2], [arr1, arr2]):
            fits.PrimaryHDU(data=a).writeto(f)

        hduls, arrs = FitsHelper.get_files([dummy_file1, dummy_file2])

        os.remove(dummy_file1)
        os.remove(dummy_file2)

        assert np.array_equal(arr1, arrs[0])
        assert np.array_equal(arr2, arrs[1])

    @staticmethod
    def test_create_mean_var_files():
        """Tests create_mean_var_files method."""
        morphs = [
            'spheroid',
            'disk',
            'irregular',
            'point_source',
            'background'
        ]
        
        local = os.path.dirname(os.path.abspath(__file__))

        shape = (200, 200)
        expected_file_names = []

        for m in morphs:
            for t in ['mean', 'var']:
                expected_file_names.append(os.path.join(local, f'{m}_{t}.fits'))

        FitsHelper.create_mean_var_files(shape, local)

        for f in expected_file_names:
            arr = fits.getdata(f)
            assert arr.shape==shape
            assert np.issubdtype(np.float32, arr.dtype)
            os.remove(f)

    @staticmethod
    def test_create_rank_vote_files():
        """Tests create_rank_vote_files method."""
        morphs = [
            'spheroid',
            'disk',
            'irregular',
            'point_source',
            'background'
        ]
        
        local = os.path.dirname(os.path.abspath(__file__))

        shape = (200, 200)
        expected_file_names = []

        for m in morphs:
            expected_file_names.append(os.path.join(local, f'{m}.fits'))

        FitsHelper.create_rank_vote_files(shape, local)

        for f in expected_file_names:
            arr = fits.getdata(f)
            assert arr.shape==shape
            assert np.issubdtype(np.float32, arr.dtype)
            os.remove(f)



