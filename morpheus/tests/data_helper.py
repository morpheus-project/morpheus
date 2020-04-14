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


def get_expected_morpheus_output(out_type="rank_vote"):
    """Gets the expected morpheus outputs for testing."""

    rank_vote_spheroid = "1nlGqibesE1LnEEif0oj-RO8xR4qNAFnP"
    rank_vote_disk = "1btsoZZJu9qWkVn0rzIK9emgdMe6mMcHS"
    rank_vote_irregular = "1qtXphVp7VflFBDWFjn6AJdvI1j3sN4KR"
    rank_vote_point_source = "16bFNlZvD_EmAMSpCU-DZ_Shq3FMpXgp3"
    rank_vote_background = "1xp6NC00T3JykdOwz0c8EFeJG0vdsThSW"
    rank_vote_n = "1I5IasDPGyDmMaXN4NCwLh27X_OuxYiPv"

    mean_spheroid = "1kXpXJTwHyAcXkC7LTLi0J8kYtViLt6eo"
    var_spheroid = "1WJ9sphwkUM-fG2XOLjDbK6cQ7akjdP_N"
    mean_disk = "1c9Sbp5IeVIosNPVIbigKfFK9gHpCR7QW"
    var_disk = "16MW0-w22827qGeHchUQd38C3FQD7WUmx"
    mean_irregular = "1RRZ9pysoJRBA_SBkfxIU625Wydo740I-"
    var_irregular = "1smaHNzdPv-FzKAn0LhalakSi5e56hx75"
    mean_point_source = "1PQUy7hNziP0koQDOR9BHk-lhz0WbqA6s"
    var_point_source = "1vdrmyrT0Io2XvhL3FCpKOHSWzQ7vlb9U"
    mean_background = "18Vi6IQHyWbd5v9BBk9Ak34V5XHQt4KtI"
    var_background = "1lCfccgqS9kbynfZBCkEl-ED7mMUtGhN5"

    if out_type == "rank_vote":
        return {
            "spheroid": get_fits(rank_vote_spheroid),
            "disk": get_fits(rank_vote_disk),
            "irregular": get_fits(rank_vote_irregular),
            "point_source": get_fits(rank_vote_point_source),
            "background": get_fits(rank_vote_background),
            "n": get_fits(rank_vote_n),
        }
    else:  # mean_var
        return {
            "spheroid_mean": get_fits(mean_spheroid),
            "spheroid_var": get_fits(var_spheroid),
            "disk_mean": get_fits(mean_disk),
            "disk_var": get_fits(var_disk),
            "irregular_mean": get_fits(mean_irregular),
            "irregular_var": get_fits(var_irregular),
            "point_source_mean": get_fits(mean_point_source),
            "point_source_var": get_fits(var_point_source),
            "background_mean": get_fits(mean_background),
            "background_var": get_fits(var_background),
            "n": get_fits(rank_vote_n),
        }


def get_expected_catalog():
    """Gets the expected catalog for testing."""

    catalog = "1UIFezBseVTz_fd9aeXopVUI8-cAyO47z"

    return {"catalog": get_json(catalog)}
