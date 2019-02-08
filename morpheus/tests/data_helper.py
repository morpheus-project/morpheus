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
"""Gets data for testing from Google Drive."""

import requests

import imageio
from astropy.io import fits

drive_url = lambda _id: f"https://drive.google.com/uc?export=download&id={_id}"
get_fits = lambda _id: fits.getdata(drive_url(_id))
get_img = lambda _id: imageio.imread(drive_url(_id))
get_json = lambda _id: requests.get(drive_url(_id)).json()


def get_expected_colorized_pngs():
    """Gets the expected colorized PNGS for testing."""

    no_hidden = "1uGl6HOxVMa1L2-jU6kHhtz65hTX3-M1L"
    hidden = "18CI48-ko8msRhkvVngvBWRvxL1fH3mE0"

    return {"no_hidden": get_img(no_hidden), "hidden": get_img(hidden)}


def get_expected_segmap():
    """Gets the expected segmentation map for testing."""

    segmap = "178875Cznj1FCsVrUcJFjm80hiG3aCQ_m"

    return {"segmap": get_fits(segmap)}


def get_expected_morpheus_output():
    """Gets the expected morpheus outputs for testing."""

    spheroid = "1nlGqibesE1LnEEif0oj-RO8xR4qNAFnP"
    disk = "1btsoZZJu9qWkVn0rzIK9emgdMe6mMcHS"
    irregular = "1qtXphVp7VflFBDWFjn6AJdvI1j3sN4KR"
    point_source = "16bFNlZvD_EmAMSpCU-DZ_Shq3FMpXgp3"
    background = "1xp6NC00T3JykdOwz0c8EFeJG0vdsThSW"
    n = "1I5IasDPGyDmMaXN4NCwLh27X_OuxYiPv"

    return {
        "spheroid": get_fits(spheroid),
        "disk": get_fits(disk),
        "irregular": get_fits(irregular),
        "point_source": get_fits(point_source),
        "background": get_fits(background),
        "n": get_fits(n),
    }


def get_expected_catalog():
    """Gets the expected catalog for testing."""

    catalog = "1UIFezBseVTz_fd9aeXopVUI8-cAyO47z"

    return {"catalog": get_json(catalog)}
