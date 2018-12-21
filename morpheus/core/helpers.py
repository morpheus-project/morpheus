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

from types import FunctionType

import tensorflow as tf

from colorama import init, Fore

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
        tf.logging.warn(TFLogger.LIGHTRED(msg))

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
    def tensor_shape(tensor: tf.Tensor, log_func=debug, format_str="[{}]::{}") -> None:
        """Log the the shape of tensor 't'.

        Args:
            tensor (tf.Tensor): A tensorflow Tensor
            logging_func (func): logging function to to use, default
                                tf_logger.default
            format_str (str): A string that will be passed will have .format called
                            on it and given two arguments in the following order:
                            - tensor_name
                            - tensor_shape
        Returns:
            None
        """
        log_func(format_str.format(tensor.name, tensor.shape.as_list()))


class OptionalFunc:
    """Descriptor protocol for functions that don't have to overriden.

    This is a helper class that is used to stub methods that don't have to
    be overridden.
    """

    @staticmethod
    def placeholder(*args):
        """Placeholder function used as default in __init__"""
        return args

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
