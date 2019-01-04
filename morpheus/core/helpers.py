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

            FitsHelper.create_file(f, shape, np.int16)

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
    """Class to help with label updates.
    
    Class Variables:
    UPDATE_MASK (np.ndarray): the (40, 40) integer array that indicates which
                              parts of the output of the model to include in the
                              calculations. default: innermost (30,30)
    UPDATE_MASK_N (np.ndarray): the (40, 40) integer array that indicates which
                                parts of the count 'n' to udpate. default:
                                all (40, 40)
    """

    # TODO: Find a better place for this
    MORPHOLOGIES = ["spheroid", "disk", "irregular", "point_source", "background"]

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
    def windowed_index_generator(dim0: int, dim1: int) -> Iterable[Tuple[int, int]]:
        """Creates a generator that returns window limited indicies over a 2d array.
        
        THe generator returned by this method will yield the indicies for the use
        of a sliding window of size `N_UPDATE_MASK.shape` over a 2d array with 
        the size `(dim0, dim1)`.

        Args:
            dim0 (int): The upper limit to iterate upto for the first dimension
            dim1 (int): The upper limit to iterate upto for the second dimension
        
        Returns:
            A generator that yields indicies to iterate over a 2d array with 
            shape [dim0, dim1]
        """

        window_y, window_x = LabelHelper.UPDATE_MASK_N.shape
        final_y = dim0 - window_y + 1
        final_x = dim1 - window_x + 1

        return LabelHelper.index_generator(final_y, final_x)

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

    @staticmethod
    def iterative_mean(
        n: np.ndarray, curr_mean: np.ndarray, x_n: np.ndarray, update_mask: np.ndarray
    ):
        """Calculates the mean of collection in an online fashion.

        The values are calculated using the following equation:
        http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf, eq. 4

        Args:
            n (np.ndarray): a 2d array containg the number of terms in mean so 
                            far,
            prev_mean (np.ndarray): the current calculated mean. 
            x_n (np.ndarray): the new values to add to the mean
            update_mask (np.ndarray): a 2d boolean array indicating which 
                                      indicies in the array should be updated.

        Returns:
            An array with the same shape as the curr_mean with the newly 
            calculated mean values.
        """
        n[n == 0] = 1
        return curr_mean + ((x_n - curr_mean) / n * update_mask)

    @staticmethod
    def iterative_variance(
        prev_sn: np.ndarray,
        x_n: np.ndarray,
        prev_mean: np.ndarray,
        curr_mean: np.ndarray,
        update_mask: np.ndarray,
    ):
        """The first of two methods used to calculate the variance online.

        This method specifically calculates the $S_n$ value as indicated in 
        equation 24 from:

        http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf

        Args:
            prev_sn (np.ndarray): the $S_n$ value from the previous step
            x_n (np.ndarray): the current incoming values
            prev_mean (np.ndarray): the mean that was previously calculated
            curr_mean (np.ndarray): the mean, including the current values
            update_mask (np.ndarray): a boolean mask indicating which values to
                                      update

        Returns:
            An np.ndarray containg the current value for $S_n$

        
        """
        return prev_sn + ((x_n - prev_mean) * (x_n - curr_mean) * update_mask)

    @staticmethod
    def finalize_variance(
        n: np.ndarray, curr_sn: np.ndarray, final_map: List[Tuple[int, int]]
    ):
        """The second of two methods used to calculate the variance online.

        This method calcaulates the final variance value using equation 25 from

        http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf 

        but without performing the square root.

        Args:
            n (np.ndarray): the current number of values included in the calculation
            curr_sn (np.ndarray): the current $S_n$ values 
            final_map List[(y, x)]: a list of indicies to calculate the final
                                    variance for

        Returns:
            A np.ndarray with the current $S_n$ values and variance values for
            all indicies in final_map
        """
        final_n = np.ones_like(n)
        for y, x in final_map:
            final_n[y, x] = n[y, x]

        return curr_sn / final_n

    @staticmethod
    def iterative_rank_vote(
        x_n: np.ndarray, prev_count: np.ndarray, update_mask: np.ndarray
    ):
        """Calculates the updated values for the rank vote labels for a one class.

        Args:
            x_n (np.ndarray): the current rank vote values for the class being 
                              updated
            prev_count (np.ndarray): the array containing the running totals,
                                     should be shaped as [labels, height, width]
            update_mask (np.ndarray): a boolean array indicating which values to 
                                      update

        Returns:    
            A numpy array containing the updated count values
        """
        update = np.zeros_like(prev_count)

        for i in range(update.shape[1]):
            for j in range(update.shape[2]):
                if update_mask[i, j]:
                    update[x_n[i, j], i, j] = 1

        count = prev_count + update

        return count

    @staticmethod
    def update_ns(data: dict, batch_idx: List[Tuple[int, int]], inc: int = 1) -> None:
        """Updates the n values by `inc`.

        Args:
            data (dict): a dictionary of numpy arrays containing the data
            batch_idx (List[Tuple[int, int]]): a list of indicies to update
            inc (int): the number to increment `n` by. Default=1

        Returns
            None
        """
        window_y, window_x = LabelHelper.UPDATE_MASK_N.shape
        for y, x in batch_idx:
            ys = slice(y, y + window_y)
            xs = slice(x, x + window_x)

            ns = data["n"][ys, xs]
            n_update = LabelHelper.UPDATE_MASK_N * LabelHelper.UPDATE_MASK * inc
            ns = ns + n_update
            data["n"][ys, xs] = ns

    @staticmethod
    def update_mean_var(
        data: dict, labels: np.ndarray, batch_idx: List[Tuple[int, int]]
    ):
        """Updates the mean and variance outputs with the new model values.
        
        Args:
            data (dict): a dict of numpy arrays containing the data
            labels (np.ndarray): the new output from the model
            batch_idx (List[Tuple[int, int]]): a list of indicies to update

        Returns:
            None
        """

        window_y, window_x = LabelHelper.UPDATE_MASK_N.shape
        total_shape = data["n"].shape
        for i, l in enumerate(labels):
            y, x = batch_idx[i]
            ys = slice(y, y + window_y)
            xs = slice(x, x + window_x)

            final_map = LabelHelper.get_final_map(total_shape, y, x)
            n = data["n"][ys, xs]
            for j, morph in enumerate(LabelHelper.MORPHOLOGIES):
                k_mean = f"{morph}_mean"
                k_var = f"{morph}_var"

                x_n = l[:, :, j]
                prev_mean = data[k_mean][ys, xs]
                prev_var = data[k_var][ys, xs]

                mean = LabelHelper.iterative_mean(
                    n, prev_mean, x_n, LabelHelper.UPDATE_MASK
                )

                var = LabelHelper.iterative_variance(
                    prev_var, x_n, prev_mean, mean, LabelHelper.UPDATE_MASK
                )
                var = LabelHelper.finalize_variance(n, var, final_map)

                data[k_mean][ys, xs] = mean
                data[k_var][ys, xs] = var

    @staticmethod
    def update_rank_vote(
        data: dict, labels: np.ndarray, batch_idx: List[Tuple[int, int]]
    ):
        """Updates the rank vote values with the new output.     

        Args:
            data (dict): data (dict): a dict of numpy arrays containing the data
            labels (np.ndarray): the new output from the model
            batch_idx (List[Tuple[int, int]]): a list of indicies to update

        Returns:
            None
        """

        window_y, window_x = LabelHelper.UPDATE_MASK_N.shape
        for i, l in enumerate(labels):
            y, x = batch_idx[i]
            ys = slice(y, y + window_y)
            xs = slice(x, x + window_x)

            ranked = l.argsort().argsort()
            for morph in enumerate(LabelHelper.MORPHOLOGIES):
                prev_count = data[morph][:, ys, xs]

                count = LabelHelper.iterative_rank_vote(
                    ranked, prev_count, LabelHelper.UPDATE_MASK
                )

                data[morph][:, ys, xs] = count

    @staticmethod
    def update_labels(
        data: dict, labels: np.ndarray, batch_idx: List[Tuple[int, int]], out_type: str
    ) -> None:
        """Updates the running total label values with the new output values.

        Args:
            data (dict): data (dict): a dict of numpy arrays containing the data
            labels (np.ndarray): the new output from the model
            batch_idx (List[Tuple[int, int]]): a list of indicies to update
            out_type (str): indicates which type of output to update must be 
                            one of ['mean_var', 'rank_vote', 'both']

        Returns:
            None
        """

        LabelHelper.update_ns(data, batch_idx)

        if out_type in ["both", "mean_var"]:
            LabelHelper.update_mean_var(data, labels, batch_idx)
        if out_type in ["both", "rank_vote"]:
            LabelHelper.update_rank_vote(data, labels, batch_idx)

    @staticmethod
    def make_mean_var_arrays(shape: Tuple[int, int]) -> dict:
        """Create output arrays for use in in-memory classification.

        Args:
            shape (Tuple[int]): The 2d (width, height) for to create the arrays

        Returns
            A dictionary with keys being the arrays description and values being
            the array itself
        """

        arrays = {}

        for morph in LabelHelper.MORPHOLOGIES:
            for t in ["mean", "var"]:
                arrays[f"{morph}_{t}"] = np.zeros(shape, dtype=np.float32)

        return arrays

    @staticmethod
    def make_rank_vote_arrays(shape: Tuple[int, int]) -> dict:
        """Create output arrays for use in in-memory classification.

        Args:
            shape (Tuple[int]): The 2d (width, height) for to create the arrays

        Returns
            A dictionary with keys being the arrays description and values being
            the array itself
        """
        shape = [5, shape[0], shape[1]]
        arrays = {}

        for morph in LabelHelper.MORPHOLOGIES:
            arrays[morph] = np.zeros(shape, dtype=np.int16)

        return arrays

    @staticmethod
    def make_n_array(shape: Tuple[int, int]) -> dict:
        """Create output array for use in in-memory classification.

        Args:
            shape (Tuple[int]): The 2d (width, height) for to create the arrays

        Returns
            A dictionary with keys being the arrays description and values being
            the array itself
        """
        return {"n": np.zeros(shape, dtype=np.int16)}
