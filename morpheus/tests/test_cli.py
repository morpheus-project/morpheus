# MIT License
# Copyright 2019 Ryan Hausen
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
"""Unit tests for the cli interface."""

import os
import pytest
import morpheus.__main__ as cli


@pytest.mark.unit
class TestCLI:
    """Tests for the cli interface."""

    @staticmethod
    def test_valid_file():
        """Tests _valid_file."""
        local = os.path.dirname(os.path.abspath(__file__))

        f_name = "test.fits"
        open(os.path.join(local, f_name), "w").close()

        valid_file = cli._valid_file(os.path.join(local, f_name))

        os.remove(os.path.join(local, f_name))

        assert valid_file

    @staticmethod
    def test_valid_file_raises():
        """Tests _valid_file, raises ValueError for incorrect file ending"""
        with pytest.raises(ValueError):
            cli._valid_file(__file__)

    @staticmethod
    def test_valid_dir():
        """Tests _valid_dir."""
        local = os.path.dirname(os.path.abspath(__file__))

        assert cli._valid_dir(local)

    @staticmethod
    def test_valid_dir_raises():
        """Tests _valid_dir raises for non dir."""
        with pytest.raises(ValueError):
            assert cli._valid_dir(__file__)

    @staticmethod
    def test_gpus():
        """Tests _gpus."""
        gpus = "1,2,3"

        assert [1, 2, 3] == cli._gpus(gpus)

    @staticmethod
    def test_gpus_raises():
        """Test _gpus raises ValueError for passing single gpus."""
        gpus = "1"

        with pytest.raises(ValueError):
            cli._gpus(gpus)

    @staticmethod
    def test_parse_args_raises_cpus_gpus():
        """test _parse_args raise ValueError for passing cpus and gpus."""

        local = os.path.dirname(os.path.abspath(__file__))
        f_name = "test.fits"
        open(os.path.join(local, f_name), "w").close()

        test_file = os.path.join(local, f_name)
        cli_args = f"{test_file} {test_file} {test_file} {test_file} "
        cli_args += "--cpus 3 --gpus 1,2,3"
        print(cli_args)
        with pytest.raises(ValueError):
            cli._parse_args(cli_args.split())

        os.remove(test_file)

    @staticmethod
    def test_parse_args_doesnt_raise():
        """test _parse_args smooth sailing."""

        local = os.path.dirname(os.path.abspath(__file__))
        f_name = "test.fits"
        open(os.path.join(local, f_name), "w").close()

        test_file = os.path.join(local, f_name)
        cli_args = f"{test_file} {test_file} {test_file} {test_file} "
        cli_args += "--cpus 3 --gpus 1,2,3"

        with pytest.raises(ValueError):
            cli._parse_args(cli_args.split())

        os.remove(test_file)
