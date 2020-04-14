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
"""Used to fetch data files"""

import os
from typing import List
from astropy.io import fits
import numpy as np


def get_sample(out_dir: str = None) -> List[np.ndarray]:
    """Retrieves a sample of CANDELS data as an example for classification.

    Sample image is taken from the CANDELS 1.0 release. The object in the center
    is GDS_deep2_3622 from Karaltepe et. al (2015).

    Data can be manually downloaded from:

    https://drive.google.com/uc?export=download&id=1fFGUVOVMOGLG4ptgAZv0T9Woimr653kn

    Args:
        out_dir (str): if not None the location to save the data at.

    Returns:
        A list of numpy arrays containing the H, J, V, and Z bands in that order

    """
    # Got direct link format from: https://www.labnol.org/internet/direct-links-for-google-drive/28356/
    url = "https://drive.google.com/uc?export=download&id=1fFGUVOVMOGLG4ptgAZv0T9Woimr653kn"

    raw_data = fits.getdata(url)
    bands = [raw_data[i] for i in range(4)]

    if out_dir:
        for name, data in zip(["h", "j", "v", "z"], bands):
            out_path = os.path.join(out_dir, f"{name}.fits")
            fits.PrimaryHDU(data=data).writeto(out_path)

    return bands
