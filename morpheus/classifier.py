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
import os
import time
from subprocess import Popen
from typing import List
from typing import Tuple
from typing import Iterable

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from astropy.io import fits

import morpheus.core.helpers as helpers
import morpheus.core.model as model


class Classifier:
    """Primary interface for the use of Morpheus.

    """

    __graph = None
    __session = None
    __X = tf.placeholder(tf.float32, shape=[None, 40, 40, 4])

    @staticmethod
    def classify_files(
        h: str = None,
        j: str = None,
        z: str = None,
        v: str = None,
        out_dir: str = ".",
        pad: bool = False,
        batch_size: int = 1000,
        out_type: str = "rank_vote",
        gpus: List[int] = None,
        parallel_check_interval: int = 15,
    ) -> None:
        """Classify FITS files using morpheus.

        Args:
            h (str): The file location of the H band FITS file
            j (str): The file location of the J band FITS file
            z (str): The file location of the Z band FITS file
            v (str): The file location of the V band FITS file
            out_dir (str): The location where to save the output files
                           if None returns the output in memory only. (None
                           not implemented)
            pad (bool): if True pad the input with zeros, so that every pixel is
                        classified the same number of times. If False, don't pad.
                        (Not implemented yet)
            batch_size (int): the number of image sections to process at a time
            out_type (str): how to process the output from Morpheus. If
                            'mean_var' record output using mean and variance, If
                            'rank_vote' record output as the normaized vote
                            count. If 'both' record both outputs.
            gpus (List[int]): A list of the CUDA gpu ID's to use for a
                              parallel classification.
            parallel_check_interval (int): If gpus are given, then this is the number
                                           of minutes to wait between polling each
                                           subprocess for completetion

        Returns:
            None
        """

        hduls, [h, j, v, z] = Classifier._parse_files(h, j, v, z)

        is_serial_run = (gpus is None) or (not gpus) or (len(gpus) == 1)
        if is_serial_run:
            Classifier.classify_arrays(
                h=h,
                j=j,
                v=v,
                z=z,
                out_type=out_type,
                pad=pad,
                out_dir=out_dir,
                batch_size=batch_size,
            )
        else:
            Classifier._build_parallel_classification_structure(
                [h, j, v, z], gpus, out_dir
            )
            Classifier._run_parallel_jobs(gpus, out_dir, parallel_check_interval)
            Classifier._stitch_parallel_classifications(out_dir)

        for hdul in hduls:
            hdul.close()

    @staticmethod
    def classify_arrays(
        h: np.ndarray = None,
        j: np.ndarray = None,
        z: np.ndarray = None,
        v: np.ndarray = None,
        out_dir: str = ".",
        pad: bool = False,
        batch_size: int = 1000,
        out_type: str = "rank_vote",
    ):
        """Classify numpy arrays using Morpheus.

        Args:
            h (np.ndarray): the H band values for an image
            j (np.ndarray): the J band values for an image
            z (np.ndarray): the Z band values for an image
            v (np.ndarray): the V band values for an iamge
            out_dir (str): The location where to save the output files
                           if None returns the output in memory only. (None
                           not implemented)
            pad (bool): if True pad the input with zeros, so that every pixel is
                        classified the same number of times. If False, don't pad.
                        (Not implemented)
            batch_size (int): the number of image sections to process at a time
            out_type (str): how to process the output from Morpheus. If
                            'mean_var' record output using mean and variance, If
                            'rank_vote' record output as the normaized vote
                            count. If 'both' record both outputs.

        Returns:
            The classification output of the model if out_dir is None(not implemented)
            otherwise None

        Raises:
            ValueError if out_type is not one of ['mean_var', 'rank_vote', 'both']
        """
        Classifier._variables_not_none(["h", "j", "z", "v"], [h, j, z, v])
        Classifier._arrays_same_size([h, j, z, v])

        if out_type not in ["mean_var", "rank_vote", "both"]:
            raise ValueError("Invalid value for `out_type`")

        if pad:
            raise NotImplementedError("pad=True has not been implemented yet")

        shape = h.shape

        hduls = []
        data = {}
        if out_dir:
            if out_type in ["mean_var", "both"]:
                hs, ds = helpers.FitsHelper.create_mean_var_files(shape, out_dir)
                hduls.extend(hs)
                data.update(ds)
            if out_type in ["rank_vote", "both"]:
                hs, ds = helpers.FitsHelper.create_rank_vote_files(shape, out_dir)
                hduls.extend(hs)
                data.update(ds)

            hs, ds = helpers.FitsHelper.create_n_file(shape, out_dir)
            hduls.extend(hs)
            ds.update(ds)
        else:
            if out_type in ["mean_var", "both"]:
                data.update(helpers.LabelHelper.make_mean_var_arrays(shape))
            if out_type in ["rank_vote", "both"]:
                data.update(helpers.LabelHelper.make_rank_vote_arrays(shape))

            data.update(helpers.LabelHelper.make_n_array(shape))

        indicies = helpers.LabelHelper.windowed_index_generator(*shape)

        window_y, window_x = helpers.LabelHelper.UPDATE_MASK_N.shape
        batch_estimate = shape[0] - window_y + 1
        batch_estimate *= shape[1] - window_x + 1
        batch_estimate = batch_estimate // batch_size
        pbar = tqdm(total=batch_estimate, desc="classifying", unit="batch")

        while True:
            batch = []
            batch_idx = []

            for _ in range(batch_size):
                try:
                    y, x = next(indicies)
                except StopIteration:
                    break

            combined = [img[y : y + window_y, x : x + window_x] for img in [h, j, v, z]]
            batch.append(Classifier._standardize_img(combined))
            batch_idx.append((y, x))

            if not batch:
                break

            batch = np.array(batch)

            labels = Classifier._call_morpheus(batch)

            helpers.LabelHelper.update_labels(data, labels, batch_idx, out_type)

            pbar.update()

        for hdul in hduls:
            hdul.close()

    @staticmethod
    def _standardize_img(img: np.ndarray) -> np.ndarray:
        """Standardizes an input img to mean 0 and unit variance.

        Uses the formula described in:

        https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization

        Args:
            img (np.ndarray): the input array to standardize

        Returns:
            The standardized input
        """
        num = img - img.mean()
        denom = max(img.std(), 1 / np.sqrt(np.prod(img.shape)))
        return num / denom

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

    @staticmethod
    def _parse_files(
        h: str, j: str, v: str, z: str
    ) -> Tuple[List[fits.HDUList], List[np.ndarray]]:
        """Validates that files exist. And returns the corresponding arrays.

        Args:
            h (str): the file location of the H band img
            j (str): the file location of the J band img
            v (str): the file location of the V band img
            z (str): the file location of the Z bnad img

        Returns:
            A tuple comtaining the a (List[HDUL], List[np.ndarray])

        Raises:
            ValueError if a variable is None

        """
        Classifier._variables_not_none(["h", "j", "z", "v"], [h, j, z, v])

        return helpers.FitsHelper.get_files([h, j, v, z])

    @staticmethod
    def _call_morpheus(batch: np.ndarray) -> np.ndarray:
        """Use morpheus to classify a batch of input values.

        Morpheus is called as a singleton using this method.

        Args:
            batch (np.ndarray): The input data in the shape
                                [batch, channels, width, height]

        Returns:
            The classified numpy array with shape [batch, width, height, channels]

        """
        batch = np.transpose(batch, axes=[0, 2, 3, 1])

        if Classifier.__graph is None:
            config = model.Morpheus.inference_hparams()
            inference_dataset = model.Morpheus.mock_dataset()

            # build graph
            m = model.Morpheus(config, inference_dataset, "channels_last")
            Classifier.__graph = m.inference(Classifier.__X)

            # get weights
            saver = tf.train.Saver()
            Classifier.__session = tf.Session()
            w_location = model.Morpheus.get_weights_dir()
            saver.restore(Classifier.__session, tf.train.latest_checkpoint(w_location))

        return Classifier.__session.run(
            Classifier.__graph, feed_dict={Classifier.__X: batch}
        )

    @staticmethod
    def _get_split_length(shape: List[int], num_gpus: int) -> int:
        """Calculate the size of the sub images for classification.

        Args:
            shape (List[int]): the shape of the array to be split
            num_gpus (int): the number of splits to make

        Returns:
            The length of each split along axis 0

        TODO: Implement splits along other axes
        """

        return (shape[0] + (num_gpus - 1) * 40) // num_gpus

    @staticmethod
    def _get_split_slice_generator(
        shape: Tuple[int], num_gpus: int, slice_length: int
    ) -> Iterable[slice]:
        """Creates a generator that yields `slice` objects to split imgs.

        Args:
            shape (Tuple[int]): The shape of the array to be split
            num_gpus (int): The number of splits to make
            split_length (int): The length each slice should be

        Returns
            A generator that yields slice objects

        TODO: Implement splits along other axes
        """

        idx = 0
        for i in range(num_gpus):
            start_idx = max(idx - 40, 0)

            if i == num_gpus - 1:
                end_idx = shape[0]
            else:
                end_idx = start_idx + slice_length

            idx = end_idx

            yield slice(start_idx, end_idx)

    @staticmethod
    def _make_runnable_file(path: str) -> None:
        """Creates a file at `path` that classfies local FITS files.

        Args:
            path (str): The dir to save the file in

        Returns:
            None
        """

        local = os.path.dirname(os.path.abspath(__file__))
        text = [
            "import sys",
            f"sys.path.append({local})",
            "import os",
            "import numpy as np",
            "from tqdm import tqdm",
            "from inference import Classifier",
            "def main():",
            "    data_dir = '.'",
            "    output_dir = './output'",
            "    if 'output' not in os.listdir():",
            "        os.mkdir('./output')",
            "    files = {",
            "        'h':os.path.join(data_dir, 'h.fits'),",
            "        'j':os.path.join(data_dir, 'j.fits'),",
            "        'v':os.path.join(data_dir, 'v.fits'),",
            "        'z':os.path.join(data_dir, 'z.fits')",
            "    }",
            "    Classifier.classify_files(h=files['h'],",
            "                              j=files['j'],",
            "                              v=files['v'],",
            "                              z=files['z'],",
            "                              batch_size=2000,",
            "                              out_type='rank_vote',",
            "                              out_dir=output_dir)",
            "if __name__=='__main__':",
            "    main()",
        ]

        with open(os.path.join(path, "main.py"), "w") as f:
            f.write("\n".join(text))

    @staticmethod
    def _stitch_parallel_classifications(out_dir: str) -> None:
        """Stitch the seperate outputs made from the parallel classifications.

        Args:
            out_dir (str): the location that contains the parallel classified
                           subdirs

        Returns:
            None
        """

        for f in helpers.LabelHelper.MORPHOLOGIES:
            to_be_stitched = []
            for output in sorted(os.listdir(out_dir)):
                if os.path.isdir(output):
                    fname = os.path.join(output, "output/{}.fits".format(f))
                    to_be_stitched.append(fits.getdata(fname)[-1, :, :])

            size = to_be_stitched[0].shape
            new_y = sum(t.shape[0] for t in to_be_stitched) - (
                40 * (len(to_be_stitched) - 1)
            )
            new_x = size[1]
            combined = np.zeros(shape=[new_y, new_x], dtype=np.float32)
            start_y = 0
            for t in to_be_stitched:
                combined[start_y : start_y + t.shape[0], :] += t
                start_y = start_y + t.shape[0] - 40

            fits.PrimaryHDU(data=combined).writeto(
                os.path.join(out_dir, f"{f}.fits"), overwrite=True
            )

    @staticmethod
    def _build_parallel_classification_structure(
        arrs: List[np.ndarray], gpus: List[int], out_dir: str
    ) -> None:
        """Sets up the subdirs and files to run the parallel classification.

        Args:
            arrs (List[np.ndarray]): List of arrays to split up in the order HJVZ
            gpus (List[int]): A list of the CUDA gpu ID's to use for a
                              parallel classification.
            out_dir (str): the location to place the subdirs in

        Returns:
            None
        """

        shape = arrs[0].shape
        num_gpus = len(gpus)
        split_slices = Classifier._get_split_slice_generator(
            shape, num_gpus, Classifier._get_split_length(shape, num_gpus)
        )

        for gpu, split_slice in tqdm(zip(sorted(gpus), split_slices)):
            sub_output_dir = os.path.join(out_dir, str(gpu))
            os.mkdir(sub_output_dir)

            for name, data in zip(["h", "j", "v", "z"], arrs):
                tmp_location = os.path.join(sub_output_dir, "{}.fits".format(name))
                fits.PrimaryHDU(data=data[split_slice, :]).writeto(tmp_location)

            Classifier._make_runnable_file(sub_output_dir)

    @staticmethod
    def _run_parallel_jobs(
        gpus: List[int], out_dir: str, parallel_check_interval: int
    ) -> None:
        """Starts and tracks parallel job runs.

        Warning: This will not finish running until all subprocesses are complete

        Args:
            gpus (List[int]): A list of the CUDA gpu ID's to use for a
                              parallel classification.
            out_dir (str): the location with the partitioned data
            parallel_check_interval (int): If gpus are given, then this is the number
                                           of minutes to wait between polling each
                                           subprocess for completetion

        Returns:
            None
        """

        processes = {}

        for gpu in gpus:
            cmd_string = f"CUDA_VISIBLE_DEVICES={gpu} python main.py"
            sub_dir = os.path.join(out_dir, gpu)
            processes[gpu] = Popen(cmd_string, shell=True, cwd=sub_dir)

        is_running = np.ones([len(gpus)], dtype=np.bool)

        while is_running.any():
            for i, g in enumerate(sorted(gpus)):
                if is_running[i] and processes[g].poll():
                    is_running[i] = False

            time.sleep(parallel_check_interval * 60)
