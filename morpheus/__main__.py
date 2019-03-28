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
"""Morpheus -- a package for making pixel level morphological classifications."""
import os
import argparse

from morpheus.classifier import Classifier


def __valid_file(value):
    if os.path.isfile(value) and value.endswith((".fits", ".FITS")):
        return value

    raise ValueError("File needs to be a fits file, ending with .fits or .FITS")


def __valid_dir(value):
    if os.path.isdir(value):
        return value

    raise ValueError("Value needs to be a directory.")


def __parse_args():
    """A place to set the arugments used in main."""
    parser = argparse.ArgumentParser(description=__doc__)

    # required arguments
    help_str = "The {} band FITS file location"
    parser.add_argument("h", type=__valid_file, help=help_str.format("H"))
    parser.add_argument("j", type=__valid_file, help=help_str.format("J"))
    parser.add_argument("v", type=__valid_file, help=help_str.format("V"))
    parser.add_argument("z", type=__valid_file, help=help_str.format("Z"))

    # optional arguments

    help_desc = "An additional optional flag to include additional information "
    help_desc += "to the morphological classifications. `catalog` saves a "
    help_desc += "morpholgical catalog as catalog.txt. `segmap` saves a "
    help_desc += "segmentation map as segmap.fits. `colorize` saves a colorized "
    help_desc += "version of the classification as colorized.png. "
    help_desc += "`all` saves catalog, segmap, and colorize."

    parser.add_argument(
        "--action",
        choices=["catalog", "segmap", "colorize", "all"],
        default="None",
        help=help_desc,
    )

    parser.add_argument("--out_dir", type=__valid_dir, default=os.getcwd())

    return parser.parse_args()


def main():
    args = __parse_args()

    if args.action == "None":
        Classifier.classify_files(
            h=args.h, j=args.j, z=args.z, v=args.v, out_dir=args.out_dir
        )
    elif args.action == "catalog":
        pass
    elif args.action == "segmap":
        pass
    elif args.action == "colorize":
        pass
    elif args.action == "all":
        pass


if __name__ == "__main__":
    main()
