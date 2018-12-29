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

"""An interface for interacting with Morpheus"""

from typing import List

import numpy as np


class Classifier:
    """Primary interface for the use of Morpheus.

    """

    @staticmethod
    def _arrays_same_size(arrays: List[np.ndarray]) -> None:
        """Verifies that all arrays are the same shape.

        Args:
            arrays (List[np.ndarray]): List of arrays that should have the same
                                       shape.

        Returns:
            None

        Raises:
            ValueError if arrays are not the same shape
        """

        arr_shapes = [a.shape for a in arrays]

        arr_comp = arr_shapes[0]
        arr_to_comp = arr_shapes[1:]

        if not np.array_equiv(arr_comp, arr_to_comp):
            raise ValueError(f"All shapes not the same: {arr_shapes}.")

    @staticmethod
    def _variables_not_none(names: List[str], values: List[np.ndarray]) -> None:
        """Verifies that all variables are not None.

        Args:
            names (List[str]): list of names of variables in the same order as 
                               `values`
            names (List[np.ndarray]): list of numpy arrays that should not be
                                      None

        Returns:
            None

        Raises:
            ValueError if a variable is None

        """

        nones = []
        for name, value in zip(names, values):
            if value is None:
                nones.append(name)

        if nones:
            raise ValueError("{} should not be None".format(nones))
