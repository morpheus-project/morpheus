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
from typing import Iterable, List, Tuple, Callable, Dict

import imageio
import numpy as np
import tensorflow as tf
from astropy.io import fits
from matplotlib.colors import hsv_to_rgb
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import sobel
from skimage.measure import regionprops
from skimage.morphology import watershed
from tqdm import tqdm

import morpheus.core.helpers as helpers
import morpheus.core.model as model


class Classifier:
    """Primary interface for the use of Morpheus.

    Images can be classified by calling
    :py:meth:`~morpheus.classifier.Classifier.classify_arrays` and passing
    numpy arrays or by calling
    :py:meth:`~morpheus.classifier.Classifier.classify_files` and passing string
    FITS file locations.

    After an image this this class offers some post processing functionality by
    generating segmentation maps using
    :py:meth:`~morpheus.classifier.Classifier.make_segmap` and colorized
    morphological classifications using
    :py:meth:`~morpheus.classifier.Classifier.colorize_rank_vote_output`

    For more examples see the `documentation <https://morpheus-astro.readthedocs.io/>`_.
    """

    __graph = None
    __session = None
    __X = tf.placeholder(tf.float32, shape=[None, 40, 40, 4])

    @staticmethod
    def catalog_arrays(
        h: str = None,
        j: str = None,
        z: str = None,
        v: str = None,
        out_dir: str = None,
        pad: bool = False,
        batch_size: int = 1000,
        out_type: str = "rank_vote",
        segmap_mask: np.ndarray = None,
        return_segmap: bool = False,
        return_morphological_map: bool = False,
    ) -> Dict:
        """Generate a source catalog using Morpheus.

        This method generates a catalog of the sources in the array. It is a
        convenience method that combines the other methods into a single action.
        The following operations are performed:

        1. The input image is classified using
           :py:meth:`~morpheus.classifier.Classifier.classify_arrays`
        2. The classification is used to generate a segmentation map using
           :py:meth:`~morpheus.classifier.Classifier.make_segmap`
        3. The the segmentation maps and the classification output are combined
           to create aggregate morphological classfifications for each source
           in the segmentation map


        Args:
            h (np.ndarray): the H band values for an image
            j (np.ndarray): the J band values for an image
            z (np.ndarray): the Z band values for an image
            v (np.ndarray): the V band values for an iamge
            out_dir (str): the location where to save the output files
                           if None returns the output in memory only.
            pad (bool): if True pad the input with zeros, so that every pixel is
                        classified the same number of times. If False, don't pad.
                        (Not implemented)
            batch_size (int): the number of image sections to process at a time
            out_type (str): how to process the output from Morpheus. If
                            'mean_var' record output using mean and variance, If
                            'rank_vote' record output as the normalized vote
                            count. If 'both' record both outputs.
            segmap_mask (np.ndarray): a boolean mask that indicates all valid
                                      pixels in the images. If None, all pixels
                                      are considered valid.
            return_segmap (bool): if `True` include the segmentation map in the
                                  returned dictionary with the key `segmentation_map`
            return_morphological_map (bool): if `True` include the morhological
                                             output maps for the input in the returned
                                             dictionary with the key `morphological_map`

        Returns:
            A dictionary containing the catalog of the classification output
            and the associated catalog. The dictionary contains the following
            keys:

            - 'classifier_output' The output from :py:meth:`~morpheus.classifier.Classifier.classify_arrays`
            - 'segmentation_map' The output from :py:meth:`~morpheus.classifier.Classifier.make_segmap`
            - 'catalog' A list of dictionary objects containing the catalog


        Raises:
            ValueError if out_type is not one of ['mean_var', 'rank_vote', 'both']
        """
        classified = Classifier.classify_arrays(
            h=h,
            j=j,
            v=v,
            z=z,
            out_dir=out_dir,
            pad=pad,
            batch_size=batch_size,
            out_type=out_type,
        )

        segmap = Classifier.make_segmap(
            data=classified, h=h, out_dir=out_dir, min_distance=20, mask=segmap_mask
        )

        # add catalog here
        catalog = Classifier.make_catalog(data=classified, flux=h, segmap=segmap)

        outs = {"catalog": catalog}

        if return_segmap:
            outs["segmentation_map"] = segmap
        if return_morphological_map:
            outs["morphological_map"] = classified

        return outs

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
        cpus: int = None,
        parallel_check_interval: int = 15,
    ) -> None:
        """Classify FITS files using Morpheus.

        Args:
            h (str): The file location of the H band FITS file
            j (str): The file location of the J band FITS file
            z (str): The file location of the Z band FITS file
            v (str): The file location of the V band FITS file
            out_dir (str): The location where to save the output files
            pad (bool): if True pad the input with zeros, so that every pixel is
                        classified the same number of times. If False, don't pad.
                        (Not implemented yet)
            batch_size (int): the number of image sections to process at a time
            out_type (str): how to process the output from Morpheus. If
                            'mean_var' record output using mean and variance, If
                            'rank_vote' record output as the normalized vote
                            count. If 'both' record both outputs.
            gpus (List[int]): A list of the CUDA gpu ID's to use for a
                              parallel classification. Don't indicate gpus and
                              cpus.
            cpus (int): The number of cpus to use in non-gpu parallel classification.
                        Don't indicate cpus and gpus.
            parallel_check_interval (int): If gpus are given, then this is the number
                                           of minutes to wait between polling each
                                           subprocess for completion

        Returns:
            None
        """
        hduls, [h, j, v, z] = Classifier._parse_files(h, j, v, z)

        is_serial_run = Classifier._validate_parallel_params(gpus, cpus)

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
        out_dir: str = None,
        pad: bool = False,
        batch_size: int = 1000,
        out_type: str = "rank_vote",
    ) -> Dict:
        """Classify numpy arrays using Morpheus.

        Args:
            h (np.ndarray): the H band values for an image
            j (np.ndarray): the J band values for an image
            z (np.ndarray): the Z band values for an image
            v (np.ndarray): the V band values for an iamge
            out_dir (str): The location where to save the output files
                           if None returns the output in memory only.
            pad (bool): if True pad the input with zeros, so that every pixel is
                        classified the same number of times. If False, don't pad.
                        (Not implemented)
            batch_size (int): the number of image sections to process at a time
            out_type (str): how to process the output from Morpheus. If
                            'mean_var' record output using mean and variance, If
                            'rank_vote' record output as the normalized vote
                            count. If 'both' record both outputs.

        Returns:
            A dictionary containing the output classifications.

        Raises:
            ValueError if out_type is not one of ['mean_var', 'rank_vote', 'both']
        """
        Classifier._variables_not_none(["h", "j", "z", "v"], [h, j, z, v])
        Classifier._arrays_same_size([h, j, z, v])

        if out_type not in ["mean_var", "rank_vote", "both"]:
            raise ValueError("Invalid value for `out_type`")

        mean_var = out_type in ["mean_var", "both"]
        rank_vote = out_type in ["rank_vote", "both"]

        if pad:
            raise NotImplementedError("pad=True has not been implemented yet")

        shape = h.shape

        hduls = []
        data = {}
        if out_dir:
            if mean_var:
                hs, ds = helpers.FitsHelper.create_mean_var_files(shape, out_dir)
                hduls.extend(hs)
                data.update(ds)
            if rank_vote:
                hs, ds = helpers.FitsHelper.create_rank_vote_files(shape, out_dir)
                hduls.extend(hs)
                data.update(ds)

            hs, ds = helpers.FitsHelper.create_n_file(shape, out_dir)
            hduls.extend(hs)
            ds.update(ds)
        else:
            if mean_var:
                data.update(helpers.LabelHelper.make_mean_var_arrays(shape))
            if rank_vote:
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

                combined = np.array(
                    [img[y : y + window_y, x : x + window_x] for img in [h, j, v, z]]
                )
                batch.append(Classifier._standardize_img(combined))
                batch_idx.append((y, x))

            if not batch:
                break

            batch = np.array(batch)

            labels = Classifier._call_morpheus(batch)

            helpers.LabelHelper.update_labels(data, labels, batch_idx, out_type)

            pbar.update()

        if rank_vote:
            helpers.LabelHelper.finalize_rank_vote(data)

        for hdul in hduls:
            hdul.close()

        return data

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
            A tuple containing the a (List[HDUL], List[np.ndarray])

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
        arrs: List[np.ndarray], workers: List[int], out_dir: str
    ) -> None:
        """Sets up the subdirs and files to run the parallel classification.

        Args:
            arrs (List[np.ndarray]): List of arrays to split up in the order HJVZ
            workers (List[int]): A list of worker ID's that can either be CUDA GPU
                                 ID's or a list of 
            out_dir (str): the location to place the subdirs in

        Returns:
            None
        """

        shape = arrs[0].shape
        num_workers = len(workers)
        split_slices = Classifier._get_split_slice_generator(
            shape, num_workers, Classifier._get_split_length(shape, num_workers)
        )

        for worker, split_slice in tqdm(zip(sorted(workers), split_slices)):
            sub_output_dir = os.path.join(out_dir, str(worker))
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

    @staticmethod
    def _validate_parallel_params(gpus: List[int], cpus:List[int]) -> bool:
        return True

    @staticmethod
    def colorize_rank_vote_output(
        data: dict, out_dir: str = None, hide_unclassified: bool = True
    ) -> np.ndarray:
        """Makes a color images from the rank_vote classification output.

        The colorization scheme is defined in HSV and is as follows:

        * Spheroid = Red
        * Disk = Blue
        * Irregular = Green
        * Point Source = Yellow

        The hue is set to be the color associated with the highest ranked class
        for a given pixel. The saturation is set to be difference between the
        highest ranked class and the second highest ranked class for a given
        pixel. For example if the top two classes have nearly equal values given
        by the classifier, then the saturation will be low and the pixel will
        appear more white. If the the top two classes have very different
        values, then the saturation will be high and the pixel's color will be
        vibrant and not white. The value for a pixel is set to be 1-bkg, where
        bkg is value given to the background class. If the background class has
        a high value, then the pixel will appear more black. If the background
        value is low, then the pixel will take on the color given by the hue and
        saturation values.

        Args:
            data (dict): A dictionary containing the output from Morpheus.
            out_dir (str): a path to save the image in.
            hide_unclassified (bool): If true black out the edges of the image
                                      that are unclassified. If false, show the
                                      borders as white.

        Returns:
            A [width, height, 3] array representing the RGB image.
        """
        red = 0.0  # spheroid
        blue = 0.7  # disk
        yellow = 0.18  # point source
        green = 0.3  # irregular

        shape = data["n"].shape

        colors = np.array([red, blue, green, yellow])
        morphs = np.dstack([data[i] for i in helpers.LabelHelper.MORPHOLOGIES[:-1]])
        ordered = np.argsort(-morphs, axis=-1)

        hues = np.zeros(shape)
        sats = np.zeros(shape)
        vals = 1 - data["background"]

        # the classifier doesn't return values for this area so black it out
        if hide_unclassified:
            vals[0:5, :] = 0
            vals[-5:, :] = 0
            vals[:, 0:5] = 0
            vals[:, -5:] = 0

        for i in tqdm(range(shape[0])):
            for j in range(shape[1]):
                hues[i, j] = colors[ordered[i, j, 0]]
                sats[i, j] = (
                    morphs[i, j, ordered[i, j, 0]] - morphs[i, j, ordered[i, j, 1]]
                )

        hsv = np.dstack([hues, sats, vals])
        rgb = hsv_to_rgb(hsv)

        if out_dir:
            png = (rgb * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(out_dir, "colorized.png"), png)

        return rgb

    @staticmethod
    def make_segmap(
        data: dict,
        h: np.ndarray,
        out_dir: str = None,
        min_distance: int = 20,
        mask: np.ndarray = None,
        deblend: bool = True,
    ) -> np.ndarray:
        """Generate a segmentation map from the classification output.

            Segmentation maps are generated using the background classification
            in ``data``. The segmenatation maps are generated using the
            `watershed algorithm <https://en.wikipedia.org/wiki/Watershed_(image_processing)>`_
            in conjunction with a couple heuristics for simple deblending. The
            steps to generate the segmentation map are as follows:

            **Inputs**: Background pixel classifications *b* and H band flux *h*

            **Outpus**: A segmentation map *sm*

            **Algorithm**:

            1. Initial Segmentation

            :math:`psf_r` = The radius of the PSF for the instrument used in H band

            *m* = a zero matrix, same size as *b*

            FOR :math:`m_{ij}` in *m*

                IF :math:`b_{ij}==1` THEN :math:`m_{ij}=1`

                ELSE IF :math:`b_{ij}==0` THEN :math:`m_{ij}=2`

            *s* = `sobel <https://en.wikipedia.org/wiki/Sobel_operator>`_ (*b*)

            *sm* = apply waterhshed on *s* with markers *m*

            2. Simple Deblending

            FOR EACH contiguous set of source pixels *p* in *sm* and corresponding flux set *f* in *h*

                Assign a unique id to *p*

                *mx* = :math:`\\max(f)`

                *pm* = pixels in *p* that are at least :math:`\\frac{mx}{10}` AND at least a :math:`psf_r` apart.

                IF there are more than 2 pixels *pm*

                    *pwm* = apply watershed on :math:`-1 * f` with markers *pm*

                    Assign unique ids to the contiguous sets of pixels in *pwm*

            RETURN *sm*

        Args:
            data (dict): A dictionary containing the output from morpheus.
            h (np.ndarray): The H band image that was classified.
            out_dir (str): A path to save the segmap in.
            min_distance (int): The minimum distance for deblending
            mask (np.ndarry): A boolean mask indicating which pixels
            deblend (bool): If ``True``, perform delblending as described in 2.
                            in the algorithm description. If ``False`` return
                            segmap without deblending.

        Returns:
            A np.ndarray segmentation map
        """
        bkg = data["background"]
        markers = np.zeros_like(h, dtype=np.uint8)

        print("Building Markers...")
        if mask is None:
            mask = data["n"] > 0

        is_bkg = np.logical_and(bkg == 1, mask)
        is_src = np.logical_and(bkg == 0, mask)

        markers[is_bkg] = 1
        markers[is_src] = 2

        sobel_img = sobel(bkg)

        print("Watershedding...")
        segmented = watershed(sobel_img, markers, mask=mask) - 1
        segmented[np.logical_not(mask)] = 0

        labeled, _ = ndi.label(segmented)

        labeled[np.logical_not(mask)] = -1

        if deblend:
            labeled = Classifier._deblend(labeled, h, min_distance)

        if out_dir:
            fits.PrimaryHDU(data=labeled).writeto(os.path.join(out_dir, "segmap.fits"))

        return labeled

    @staticmethod
    def _deblend(segmap: np.ndarray, flux: np.ndarray, min_distance: int) -> np.ndarray:
        """Deblends a segmentation map according to the description in make_segmap.

        Args:
            segmap (np.ndarray): The segmentation map image to deblend
            flux (np.ndarray): The corresponding flux image in H band
            min_distance (int): The radius of the PSF for the instrument used on H band

        Returns:
            A np.ndarray representing the deblended segmap
        """

        max_id = segmap.max()

        for region in tqdm(regionprops(segmap, flux), desc="Deblending"):

            # greater than 1 indicates that the region is not background
            if region.label > 0:
                flx = region.intensity_image
                seg = region.filled_image
                flux_map = flx * seg

                maxes = peak_local_max(
                    flux_map, min_distance=min_distance, num_peaks=20
                )

                # more than 1 source found, deblend
                if maxes.shape[0] > 1:
                    start_y, start_x, end_y, end_x = region.bbox
                    markers = np.zeros_like(seg, dtype=np.int)

                    for y, x in maxes:
                        max_id += 1
                        markers[y, x] = max_id

                    deblended = watershed(-flux_map, markers, mask=seg)

                    local_segmap = segmap[start_y:end_y, start_x:end_x].copy()
                    local_segmap = np.where(seg, deblended, local_segmap)
                    segmap[start_y:end_y, start_x:end_x] = local_segmap

        return segmap

    @staticmethod
    def aggregation_scheme_flux_weighted(
        data: dict, flux: np.ndarray, segmap: np.ndarray
    ) -> List[float]:
        """Aggreates pixel level morphological classifcations to the source level.

        Uses a flux weighted mean of the pixel level morphologies to calculate
        the aggregate source level morphology.

        Args:
            data (dict): A dictionary containing the output from morpheus.
            flux (np.ndarray): The corresponding flux image in H band
            segmap (int): The binary map indicating pixels that belong to the
                          source

        Returns:
            The morphological classification as a list of floats in the
            following order: ['spheroid', 'disk', 'irregular', 'point source']
        """
        classifications = np.zeros([4])

        morphs = ["spheroid", "disk", "irregular", "point_source"]

        morphs = [data[m] for m in morphs]

        for i, m in enumerate(morphs):
            classifications[i] = np.mean(m[segmap] * flux[segmap])

        return (classifications / classifications.sum()).tolist()

    @staticmethod
    def make_catalog(
        data: dict,
        flux: np.ndarray,
        segmap: np.ndarray,
        aggregation_scheme: Callable = None,
    ) -> List[Dict]:
        """Creates a catalog of sources and their morphologies.

        Args:
            data (dict): A dictionary containing the output from morpheus.
            flux (np.ndarray): The corresponding flux image in H band
            segmap (int): A labelled segmap where every pixel with a
                          value > 0 is associated with a source.
            aggregation_scheme (func): Function that takes three arguments `data`,
                                       `flux`, and `segmap`, same as to this
                                       function, then returns a numpy array
                                       containing the morphological classification
                                       in the following order-shperoid, disk,
                                       irregular, and point source/compact. If
                                       None, then the flux weighting scheme
                                       in

        Returns:
            A JSON-compatible list of dictionary objects with the following keys:
            {
                'id': the id from the segmap
                'location': a (y,x) location -- the max pixel within the segmap
                'morphology': a dictionary containing the morphology values.
            }
        """

        if aggregation_scheme is None:
            aggregation_scheme = Classifier.aggregation_scheme_flux_weighted

        catalog = []

        for region in regionprops(segmap, flux):
            _id = region.label

            if _id < 1:
                continue

            img = region.intensity_image
            seg = region.filled_image

            start_y, start_x, end_y, end_x = region.bbox
            dat = {}
            for k in data:
                dat[k] = data[k][start_y:end_y, start_x:end_x].copy()

            classification = aggregation_scheme(dat, img, seg)

            masked_flux = img * seg

            # https://stackoverflow.com/a/3584260
            local_y, local_x = np.unravel_index(masked_flux.argmax(), masked_flux.shape)
            global_y, global_x = int(start_y + local_y), int(start_x + local_x)

            catalog.append(
                {
                    "id": _id,
                    "location": [global_y, global_x],
                    "morphology": classification,
                }
            )

        return catalog
