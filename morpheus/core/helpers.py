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
"""Helper classes used in Morpheus."""

import os

from types import FunctionType
from typing import List
from typing import Iterable
from typing import Tuple

import numpy as np
import tensorflow as tf

from colorama import init, Fore
from astropy.io import fits

init(autoreset=True)


class TFLogger:
    """A helper class to color the logging text in Tensorflow."""

    RED = lambda s: Fore.RED + str(s) + Fore.RESET
    BLUE = lambda s: Fore.BLUE + str(s) + Fore.RESET
    YELLOW = lambda s: Fore.YELLOW + str(s) + Fore.RESET
    GREEN = lambda s: Fore.GREEN + str(s) + Fore.RESET
    LIGHTRED = lambda s: Fore.LIGHTRED_EX + str(s) + Fore.RESET

    @staticmethod
    def info(msg: str) -> None:
        """Log at info level in green.

        Args:
            msg (str): The string to be logged

        Returns:
            None
        """
        tf.logging.info(TFLogger.GREEN(msg))

    @staticmethod
    def debug(msg: str) -> None:
        """Log at debug level in yellow.

        Args:
            msg (str): The string to be logged

        Returns:
            None
        """
        tf.logging.debug(TFLogger.YELLOW(msg))

    @staticmethod
    def warn(msg: str) -> None:
        """Log at warn level in lightred.

        Args:
            msg (str): The string to be logged

        Returns:
            None
        """
        tf.logging.warning(TFLogger.LIGHTRED(msg))

    @staticmethod
    def error(msg: str):
        """Log at error level in red.

        Args:
            msg (str): The string to be logged

        Returns:
            None
        """
        tf.logging.error(TFLogger.RED(msg))

    @staticmethod
    def tensor_shape(tensor: tf.Tensor, log_func=None, format_str="[{}]::{}") -> None:
        """Log the the shape of tensor 't'.

        Args:
            tensor (tf.Tensor): A tensorflow Tensor
            logging_func (func): logging function to to use, default
                                tf_logger.debug
            format_str (str): A string that will be passed will have .format called
                            on it and given two arguments in the following order:
                            - tensor_name
                            - tensor_shape
        Returns:
            None
        """
        if log_func is None:
            log_func = TFLogger.debug

        log_func(format_str.format(tensor.name, tensor.shape.as_list()))


class OptionalFunc:
    """Descriptor protocol for functions that don't have to overriden.

    This is a helper class that is used to stub methods that don't have to
    be overridden.
    """

    @staticmethod
    def placeholder(*args):
        """Placeholder function used as default in __init__"""
        return list(*args)

    def __init__(self, warn_msg: str, init_func: FunctionType = placeholder):
        """"""
        self._warn_msg = warn_msg
        self._func = init_func
        self._is_default = True

    def __get__(
        self, obj, type=None  # pylint: disable=redefined-builtin
    ) -> FunctionType:
        if self._is_default:
            TFLogger.warn(self._warn_msg)

        return self._func

    def __set__(self, obj, value) -> None:
        self._is_default = False
        self._func = value


class FitsHelper:
    """A class that handles basic FITS file functions."""

    # TODO: Find a better place for this
    MORPHOLOGIES = ["spheroid", "disk", "irregular", "point_source", "background"]

    @staticmethod
    def create_file(file_name: str, data_shape: tuple, dtype) -> None:
        """Creates a fits file without loading it into memory.

        This is a helper method to create large FITS files without loading an
        array into memory. The method follows the direction given at:
        http://docs.astropy.org/en/stable/generated/examples/io/skip_create-large-fits.html


        Args:
            file_name (str): the complete path to the file to be created.
            data_shape (tuple): a tuple describe the shape of the file to be
                                created
            dtype (numpy datatype): the numpy datatype used in the array

        Raises:
            ValueError if dtype is not one of:
                - np.unit8
                - np.int16
                - np.int32
                - np.float32
                - np.float64


        TODO: Figure out why this throws warning about size occasionally
              when files that are created by it are opened
        """
        bytes_per_value = 0

        if dtype == np.uint8:
            bytes_per_value = 1
        elif dtype == np.int16:
            bytes_per_value = 2
        elif dtype == np.int32:
            bytes_per_value = 4
        elif dtype == np.float32:
            bytes_per_value = 4
        elif dtype == np.float64:
            bytes_per_value = 8

        if bytes_per_value == 0:
            raise ValueError("Invalid dtype")

        stub_size = [100, 100]
        if len(data_shape) == 3:
            stub_size.append(5)
        stub = np.zeros(stub_size, dtype=dtype)

        hdu = fits.PrimaryHDU(data=stub)
        header = hdu.header
        while len(header) < (36 * 4 - 1):
            header.append()

        header["NAXIS1"] = data_shape[1]
        header["NAXIS2"] = data_shape[0]
        if len(data_shape) == 3:
            header["NAXIS3"] = data_shape[2]

        header.tofile(file_name)

        with open(file_name, "rb+") as f:
            header_size = len(header.tostring())
            data_size = (np.prod(data_shape) * bytes_per_value) - 1

            f.seek(header_size + data_size)
            f.write(b"\0")

    @staticmethod
    def get_files(
        file_names: List[str], mode: str = "readonly"
    ) -> (List[fits.HDUList], List[np.ndarray]):
        """Gets the HDULS and data handles for all the files in file_names.

        This is a convience function to opening multiple FITS files using
        memmap.

        Args:
            file_names (List[str]): a list of file names including paths to FITS
                                    files
            mode (str): the mode to pass to fits.open

        Returns:
            Tuple of a list numpy arrays that are the mmapped data handles for
            each of the FITS files and the HDULs that go along with them
        """
        arrays = []
        hduls = []

        for f in file_names:
            hdul = fits.open(f, mode=mode, memmap=True)
            arrays.append(hdul[0].data)  # Astropy problem pylint: disable=E1101
            hduls.append(hdul)

        return hduls, arrays

    @staticmethod
    def create_mean_var_files(
        shape: List[int], out_dir: str
    ) -> (List[fits.HDUList], List[np.ndarray]):
        """Creates the output fits files for the mean/variance morpheus output.

            Args:
                shape (List[int]): The shape to use when making the FITS files
                out_dir (str): the directory to place the files in. Will make it
                               if it doesn't already exist.

            Returns:
                List[fits.HDUList]: for the created files
                Dict(str, np.ndarray): a dictionary where the key is the data 
                                       descriptor and the value is the memmapped
                                       data numpy array
        """

        data_keys = []
        file_names = []
        for morph in FitsHelper.MORPHOLOGIES:
            for t in ["mean", "var"]:
                f = os.path.join(out_dir, f"{morph}_{t}.fits")
                file_names.append(f)
                data_keys.append(f"{morph}_{t}")

                FitsHelper.create_file(f, shape, np.float32)

        hduls, arrays = FitsHelper.get_files(file_names, mode="update")

        return hduls, {k: v for k, v in zip(data_keys, arrays)}

    @staticmethod
    def create_rank_vote_files(
        shape: List[int], out_dir: str
    ) -> (List[fits.HDUList], List[np.ndarray]):
        """Creates the output fits files for the rank vote morpheus output.

            Args:
                shape (List[int]): The shape to use when making the FITS files
                out_dir (str): the directory to place the files in. Will make it
                               if it doesn't already exist.

            Returns:
                List[fits.HDUList]: for the created files
                Dict(str, np.ndarray): a dictionary where the key is the data 
                                       descriptor and the value is the memmapped
                                       data numpy array
        """

        data_keys = []
        file_names = []
        for morph in FitsHelper.MORPHOLOGIES:
            f = os.path.join(out_dir, f"{morph}.fits")
            file_names.append(f)
            data_keys.append(morph)

            FitsHelper.create_file(f, shape, np.float32)

        hduls, arrays = FitsHelper.get_files(file_names, mode="update")

        return hduls, {k: v for k, v in zip(data_keys, arrays)}

    @staticmethod
    def create_n_file(
        shape: List[int], out_dir: str
    ) -> (List[fits.HDUList], List[np.ndarray]):
        """Creates the output fits files for the rank vote morpheus output.

            Args:
                shape (List[int]): The shape to use when making the FITS files
                out_dir (str): the directory to place the files in. Will make it
                               if it doesn't already exist.

            Returns:
                List[fits.HDUList]: for the created files
                Dict(str, np.ndarray): a dictionary where the key is the data 
                                       descriptor and the value is the memmapped
                                       data numpy array
        """

        n_path = os.path.join(out_dir, "n.fits")
        FitsHelper.create_file(n_path, shape, np.int16)
        hduls, arrays = FitsHelper.get_files([n_path], mode="update")

        return hduls, {"n": arrays[0]}


class LabelHelper:
    """Class to help with label updates"""

    UPDATE_MASK = np.pad(np.ones([30, 30]), 5, mode="constant").astype(np.int16)
    UPDATE_MASK_N = np.ones([40, 40], dtype=np.int16)

    @staticmethod
    def index_generator(dim0: int, dim1: int) -> Iterable[Tuple[int, int]]:
        """Creates a generator that returns indicies to iterate over a 2d array.
        
        Args:
            dim0 (int): The upper limit to iterate upto for the first dimension
            dim1 (int): The upper limit to iterate upto for the second dimension
        
        Returns:
            A generator that yields indicies to iterate over a 2d array with 
            shape [dim0, dim1]
        """
        for y in range(dim0):
            for x in range(dim1):
                yield (y, x)

    @staticmethod
    def get_final_map(shape: List[int], y: int, x: int):
        """Creates a pixel mapping that flags pixels that won't be updated again.
        
        Args:
            shape (List[int]): the shape of the array that x and y are indexing
            y (int): the current y index
            x (int): the current x index

        Returns:
            A list of relative indicies that won't be updated again.
        """
        final_map = []

        end_y = y == (shape[0] - LabelHelper.UPDATE_MASK_N.shape[0])
        end_x = x == (shape[1] - LabelHelper.UPDATE_MASK_N.shape[1])

        if end_y and end_x:
            for _y in range(5, 35):
                for _x in range(5, 35):
                    final_map.append((_y, _x))
        else:
            if end_x:
                final_map.extend([(5, _x) for _x in range(5, 35)])
            if end_y:
                final_map.extend([(_y, 5) for _y in range(5, 35)])

        if not final_map:
            final_map.append((5, 5))

        return final_map
