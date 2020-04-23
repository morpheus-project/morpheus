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
import json
from subprocess import Popen
from typing import Iterable, List, Tuple, Callable, Dict, Union

import imageio
import numpy as np
import tensorflow.compat.v1 as tf
from astropy.io import fits
from matplotlib.colors import hsv_to_rgb
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import sobel
from skimage.measure import regionprops
from skimage.morphology import watershed
from tqdm import tqdm

tf.disable_eager_execution()

import morpheus.core.helpers as helpers
import morpheus.core.model as model


class Classifier:
    """The primary interface for the use of Morpheus.

    Images can be classified by calling
    :py:meth:`~morpheus.classifier.Classifier.classify` and passing
    numpy arrays or string FITS file locations.

    After an image this this class offers some post processing functionality by
    generating segmentation maps using
    :py:meth:`~morpheus.classifier.Classifier.segmap_from_classified`, colorized
    morphological classifications using
    :py:meth:`~morpheus.classifier.Classifier.colorize_classification`, and
    generating catalogs using
    :py:meth:`~morpheus.classifier.Classifier.catalog_from_classified`.

    For more examples, see the `documentation <https://morpheus-astro.readthedocs.io/>`_.
    """

    __graph = None
    __session = None
    __X = tf.placeholder(tf.float32, shape=[None, 40, 40, 4])

    @staticmethod
    def classify(
        h: Union[np.ndarray, str] = None,
        j: Union[np.ndarray, str] = None,
        z: Union[np.ndarray, str] = None,
        v: Union[np.ndarray, str] = None,
        out_dir: str = None,
        batch_size: int = 1000,
        out_type: str = "rank_vote",
        gpus: List[int] = None,
        cpus: int = None,
        parallel_check_interval: float = 1,
    ) -> dict:
        """Generates per-pixel classifications from input images.

        Args:
            h (Union[np.ndarray, str]): The H band image or the path to it
            j (Union[np.ndarray, str]): The J band image or the path to it
            v (Union[np.ndarray, str]): The V band image or the path to it
            z (Union[np.ndarray, str]): The Z band image or the path to it
            out_dir (str): If provided, a directory to save the output to
            batch_size (int): The size of the batches to use when classifying the input
            out_type (str): The method by which to aggregate classifications
                            for a single pixel. Can be one of "rank_vote",
                            "mean_var", or "both"
            gpus (List[int]): The GPU ids to use for parallel classification
                              the ids can be found using ``nvidia-smi``
            cpus (int): The number of cpus to use for parallel classification.
            parallel_check_interval (float): If running a parallel job, how often
                                           to check on the running sub-processes
                                           in minutes.

        Returns:
            Dictionary containing the classification output for the given input

        Raises:
            ValueError if both gpus and cpus are given
            ValueError if mixed string and numpy arrays are given for h, j, v, z
            ValueError if h, j, v, or z are None
        """

        Classifier._variables_not_none(["h", "j", "v", "z"], [h, j, v, z])
        are_files = Classifier._valid_input_types_is_str(h, j, v, z)
        workers, is_gpu = Classifier._validate_parallel_params(gpus, cpus)

        if are_files:
            hduls, [h, j, v, z] = Classifier._parse_files(h, j, v, z)

            if out_dir is None:
                out_dir = "."
        else:
            hduls = []

        if len(workers) == 1:
            classified = Classifier._classify_arrays(
                h=h,
                j=j,
                v=v,
                z=z,
                out_type=out_type,
                out_dir=out_dir,
                batch_size=batch_size,
            )
        else:
            Classifier._build_parallel_classification_structure(
                [h, j, v, z], workers, batch_size, out_dir, out_type
            )
            Classifier._run_parallel_jobs(
                workers, is_gpu, out_dir, parallel_check_interval
            )
            Classifier._stitch_parallel_classifications(workers, out_dir, out_type)

            classification_hduls, classified = Classifier._retrieve_classifications(
                out_dir, out_type
            )

            hduls.extend(classification_hduls)

        for hdul in hduls:
            hdul.close()

        return classified

    @staticmethod
    def catalog_from_classified(
        classified: dict,
        flux: np.ndarray,
        segmap: np.ndarray,
        aggregation_scheme: Callable = None,
        out_file: str = None,
    ) -> List[Dict]:
        """Creates a catalog of sources and their morphologies.

        Args:
            classified (dict): A dictionary containing the output from morpheus.
            flux (np.ndarray): The corresponding flux image in H band
            segmap (np.ndarray): A labeled segmap where every pixel with a
                          value > 0 is associated with a source.
            aggregation_scheme (func): Function that takes three arguments `classified`,
                                       `flux`, and `segmap`, same as this
                                       function, then returns a numpy array
                                       containing the morphological classification
                                       in the following order-spheroid, disk,
                                       irregular, and point source/compact. If
                                       None, then the flux weighting scheme
                                       in
            out_file (str): a location to save the catalog. Can either be .csv
                            or .json. Anything else will raise a ValueError.


        Returns:
            A JSON-compatible list of dictionary objects with the following keys:
            {
                'id': the id from the segmap
                'location': a (y,x) location -- the max pixel within the segmap
                'morphology': a dictionary containing the morphology values.
            }
        """

        if out_file:
            if out_file.endswith((".csv", ".json")):
                is_csv = out_file.endswith(".csv")
            else:
                raise ValueError("out_file must end with .csv or .json")

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
            for k in classified:
                dat[k] = classified[k][start_y:end_y, start_x:end_x].copy()

            classification = aggregation_scheme(dat, img, seg)

            masked_flux = img * seg

            # https://stackoverflow.com/a/3584260
            y, x = np.unravel_index(masked_flux.argmax(), masked_flux.shape)
            y, x = int(start_y + y), int(start_x + x)

            catalog.append(
                {"id": _id, "location": [y, x], "morphology": classification}
            )

        if out_file:
            with open(out_file, "w") as f:
                if is_csv:
                    f.write("source_id,y,x,sph,dsk,irr,ps\n")

                    for c in catalog:
                        csv = "{},{},{},{},{},{},{}\n"
                        f.write(
                            csv.format(
                                c["id"],
                                c["location"][0],
                                c["location"][1],
                                c["morphology"][0],
                                c["morphology"][1],
                                c["morphology"][2],
                                c["morphology"][3],
                            )
                        )
                else:
                    json.dump(catalog, f)

        return catalog

    # TODO: make the output file with the FITS helper if the output dir is used.
    @staticmethod
    def segmap_from_classified(
        classified: dict,
        flux: np.ndarray,
        bkg_src_threshold: float = 0.0,
        out_dir: str = None,
        min_distance: int = 20,
        mask: np.ndarray = None,
        deblend: bool = True,
    ) -> np.ndarray:
        """Generate a segmentation map from the classification output.

        For more information about the segmentation process, see:
        https://arxiv.org/abs/1906.11248

        Args:
            data (dict): A dictionary containing the output from morpheus.
            flux (np.ndarray): The flux to use when making the segmap
            bkg_src_threshold (float): The max value that a background
                                       classification pixel can take and be
                                       considered a source. The default is 0.
                                       Should be between [0,1]
            out_dir (str): A path to save the segmap in.
            min_distance (int): The minimum distance for deblending
            mask (np.ndarry): A boolean mask indicating which pixels
            deblend (bool): If ``True``, perform deblending as described in 2.
                            in the algorithm description. If ``False`` return
                            segmap without deblending.

        Returns:
            A np.ndarray segmentation map
        """
        if bkg_src_threshold < 0 or bkg_src_threshold >= 1:
            err_msg = [
                "Invalid value for `bkg_src_threshold`, use a value in the ",
                "range [0, 1)",
            ]

            raise ValueError(err_msg)

        bkg = classified["background"]
        markers = np.zeros_like(flux, dtype=np.uint8)

        print("Building Markers...")
        if mask is None:
            mask = classified["n"] > 0

        is_bkg = np.logical_and(bkg == 1, mask)
        is_src = np.logical_and(bkg <= bkg_src_threshold, mask)

        markers[is_bkg] = 1
        markers[is_src] = 2

        sobel_img = sobel(bkg)

        print("Watershedding...")
        segmented = watershed(sobel_img, markers, mask=mask) - 1
        segmented[np.logical_not(mask)] = 0

        labeled, _ = ndi.label(segmented)

        labeled[np.logical_not(mask)] = -1

        if deblend:
            labeled = Classifier._deblend(labeled, flux, min_distance)

        if out_dir:
            fits.PrimaryHDU(data=labeled).writeto(os.path.join(out_dir, "segmap.fits"))

        return labeled

    @staticmethod
    def colorize_classified(
        classified: dict, out_dir: str = None, hide_unclassified: bool = True
    ) -> np.ndarray:
        """Makes a color image from the classification output.

        The colorization scheme is defined in HSV and is as follows:

        * Spheroid = Red
        * Disk = Blue
        * Irregular = Green
        * Point Source = Yellow

        The hue is set to be the color associated with the highest ranked class
        for a given pixel. The saturation is set to be the difference between the
        highest ranked class and the second highest ranked class for a given
        pixel. For example, if the top two classes have nearly equal values given
        by the classifier, then the saturation will be low and the pixel will
        appear more white. If the top two classes have very different
        values, then the saturation will be high and the pixel's color will be
        vibrant and not white. The value for a pixel is set to be 1-bkg, where
        bkg is value given to the background class. If the background class has
        a high value, then the pixel will appear more black. If the background
        value is low, then the pixel will take on the color given by the hue and
        saturation values.

        Args:
            data (dict): A dictionary containing the output from Morpheus.
            out_dir (str): a path to save the image in.
            hide_unclassified (bool): If true, black out the edges of the image
                                      that are unclassified. If false, show the
                                      borders as white.

        Returns:
            A [width, height, 3] array representing the RGB image.
        """
        red = 0.0  # spheroid
        blue = 0.7  # disk
        yellow = 0.18  # point source
        green = 0.3  # irregular

        shape = classified["n"].shape

        colors = np.array([red, blue, green, yellow])
        morphs = np.dstack(
            [classified[i] for i in helpers.LabelHelper.MORPHOLOGIES[:-1]]
        )
        ordered = np.argsort(-morphs, axis=-1)

        hues = np.zeros(shape)
        sats = np.zeros(shape)
        vals = 1 - classified["background"]

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
    def _retrieve_classifications(
        out_dir: str, out_type: str
    ) -> Tuple[List[fits.HDUList], dict]:

        f_names = []
        for morph in helpers.LabelHelper.MORPHOLOGIES:
            if out_type in ["mean_var", "both"]:
                f_names.extend(
                    [
                        os.path.join(out_dir, f"{morph}_mean.fits"),
                        os.path.join(out_dir, f"{morph}_var.fits"),
                    ]
                )
            if out_type in ["rank_vote", "both"]:
                f_names.append(os.path.join(out_dir, f"{morph}.fits"))

        f_names.append(os.path.join(out_dir, "n.fits"))

        hduls, arrs = helpers.FitsHelper.get_files(f_names)

        classified = {
            os.path.split(n)[1].replace(".fits", ""): a for n, a in zip(f_names, arrs)
        }

        return hduls, classified

    @staticmethod
    def _valid_input_types_is_str(
        h: Union[np.ndarray, str] = None,
        j: Union[np.ndarray, str] = None,
        z: Union[np.ndarray, str] = None,
        v: Union[np.ndarray, str] = None,
    ):
        in_types = {type(val) for val in [h, j, z, v]}

        if len(in_types) > 1:
            raise ValueError(
                "Mixed input type usuage. Ensure all are numpy arrays or strings."
            )

        t = in_types.pop()

        if t in [np.ndarray, str]:
            return t == str
        else:
            raise ValueError("Input type must either be numpy array or string")

    # NEW API ==================================================================

    @staticmethod
    def _classify_arrays(
        h: np.ndarray = None,
        j: np.ndarray = None,
        z: np.ndarray = None,
        v: np.ndarray = None,
        out_dir: str = None,
        batch_size: int = 1000,
        out_type: str = "rank_vote",
    ) -> Dict:
        """Classify numpy arrays using Morpheus.

        Args:
            h (np.ndarray): the H band values for an image
            j (np.ndarray): the J band values for an image
            z (np.ndarray): the Z band values for an image
            v (np.ndarray): the V band values for an image
            out_dir (str): The location where to save the output files
                           if None returns the output in memory only.
            batch_size (int): the number of image sections blackto process at a time
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
            data.update(ds)
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
    def _get_split_length(shape: List[int], num_workers: int) -> int:
        """Calculate the size of the sub images for classification.

        Args:
            shape (List[int]): the shape of the array to be split
            num_workers (int): the number of splits to make

        Returns:
            The length of each split along axis 0

        TODO: Implement splits along other axes
        """

        return (shape[0] + (num_workers - 1) * 40) // num_workers

    @staticmethod
    def _get_split_slice_generator(
        shape: Tuple[int], num_workers: int, slice_length: int
    ) -> Iterable[slice]:
        """Creates a generator that yields `slice` objects to split imgs.

        Args:
            shape (Tuple[int]): The shape of the array to be split
            num_workers (int): The number of splits to make
            split_length (int): The length each slice should be

        Returns
            A generator that yields slice objects

        TODO: Implement splits along other axes
        """

        idx = 0
        for i in range(num_workers):
            start_idx = max(idx - 39, 0)

            if i == num_workers - 1:
                end_idx = shape[0]
            else:
                end_idx = start_idx + slice_length - 1

            idx = end_idx

            yield slice(start_idx, end_idx)

    @staticmethod
    def _make_runnable_file(
        path: str, batch_size: int = 1000, out_type: str = "rank_vote"
    ) -> None:
        """Creates a file at `path` that classfies local FITS files.

        Args:
            path (str): The dir to save the file in
            batch_size (int): The batch size for Morpheus to use when classifying
                              the input
            out_type (str): how to process the output from Morpheus. If
                            'mean_var' record output using mean and variance, If
                            'rank_vote' record output as the normalized vote
                            count. If 'both' record both outputs.

        Returns:
            None
        """

        local = os.path.dirname(os.path.dirname(__file__))
        text = [
            "import sys",
            f'sys.path.append("{local}")',
            "import os",
            "import numpy as np",
            "from tqdm import tqdm",
            "from morpheus.classifier import Classifier",
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
            "    Classifier.classify(h=files['h'],",
            "                        j=files['j'],",
            "                        v=files['v'],",
            "                        z=files['z'],",
            f"                       batch_size={batch_size},",
            f'                       out_type="{out_type}",',
            "                        out_dir=output_dir)",
            "    sys.exit(0)",
            "if __name__=='__main__':",
            "    main()",
        ]

        with open(os.path.join(path, "main.py"), "w") as f:
            f.write("\n".join(text))

    @staticmethod
    def _build_parallel_classification_structure(
        arrs: List[np.ndarray],
        workers: List[int],
        batch_size: int,
        out_dir: str,
        out_type: str,
    ) -> None:
        """Sets up the subdirs and files to run the parallel classification.

        Args:
            arrs (List[np.ndarray]): List of arrays to split up in the order HJVZ
            workers (List[int]): A list of worker ID's that can either be CUDA GPU
                                 ID's or a list dummy numbers for cpu workers
            batch_size (int): The batch size for Morpheus to use when classifying
                              the input.
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

            Classifier._make_runnable_file(sub_output_dir, batch_size, out_type)

    @staticmethod
    def _stitch_parallel_classifications(
        workers: List[int], out_dir: str, out_type: str
    ) -> None:
        """Stitch the seperate outputs made from the parallel classifications.

        Args:
            workers (List[int]): A list of worker ID's that can either be CUDA GPU
                                 ID's or a list dummy numbers for cpu workers
            out_dir (str): the location that contains the parallel classified
                           subdirs
            out_type (str): how to process the output from Morpheus. If
                            'mean_var' record output using mean and variance, If
                            'rank_vote' record output as the normalized vote
                            count. If 'both' record both outputs.

        Returns:
            None
        """
        jobs = []
        if out_type in ["mean_var", "both"]:
            jobs.append("mean_var")
        if out_type in ["rank_vote", "both"]:
            jobs.append("rank_vote")

        for morph in helpers.LabelHelper.MORPHOLOGIES:
            for job in jobs:
                if job == "mean_var":
                    to_be_stitched = []
                    for worker_id in workers:  # each worker was assinged a dir by id
                        dir_list = [out_dir, str(worker_id), "output"]
                        f_mean = os.path.join(*(dir_list + [f"{morph}_mean.fits"]))
                        f_var = os.path.join(*(dir_list + [f"{morph}_var.fits"]))
                        f_n = os.path.join(*(dir_list + ["n.fits"]))

                        to_be_stitched.append(
                            (
                                fits.getdata(f_mean),
                                fits.getdata(f_var),
                                fits.getdata(f_n),
                            )
                        )

                    new_y = sum(t[0].shape[0] for t in to_be_stitched)
                    new_y -= 39 * (len(to_be_stitched) - 1)

                    new_x = to_be_stitched[0][0].shape[1]

                    combined_mean = np.zeros(shape=[new_y, new_x], dtype=np.float32)
                    combined_var = np.zeros(shape=[new_y, new_x], dtype=np.float32)
                    combined_n = np.zeros(shape=[new_y, new_x], dtype=np.float32)

                    start_y = 0
                    for new_mean, new_var, new_n in to_be_stitched:
                        Classifier._merge_parallel_means_vars(
                            combined_mean,
                            combined_var,
                            combined_n,
                            new_mean,
                            new_var,
                            new_n,
                            start_y,
                        )

                        start_y += new_n.shape[0] - 39

                    to_write = [
                        (combined_mean, f"{morph}_mean.fits"),
                        (combined_var, f"{morph}_var.fits"),
                        (combined_n, "n.fits"),
                    ]

                    for f, n in to_write:
                        fits.PrimaryHDU(data=f).writeto(
                            os.path.join(out_dir, n), overwrite=True
                        )

                if job == "rank_vote":
                    to_be_stitched = []
                    for worker_id in workers:  # each worker was assinged a dir by id
                        dir_list = [out_dir, str(worker_id), "output"]
                        f_votes = os.path.join(*(dir_list + [f"{morph}.fits"]))
                        f_n = os.path.join(*(dir_list + ["n.fits"]))

                        to_be_stitched.append(
                            (fits.getdata(f_votes), fits.getdata(f_n))
                        )

                    new_y = sum(t[0].shape[0] for t in to_be_stitched)
                    new_y -= 39 * (len(to_be_stitched) - 1)

                    new_x = to_be_stitched[0][0].shape[1]

                    combined_votes = np.zeros(shape=[new_y, new_x], dtype=np.float32)
                    combined_n = np.zeros(shape=[new_y, new_x], dtype=np.float32)

                    start_y = 0
                    for new_votes, new_n in to_be_stitched:
                        Classifier._merge_parallel_rank_votes(
                            combined_votes, combined_n, new_votes, new_n, start_y
                        )

                        start_y += new_n.shape[0] - 39

                    to_write = [
                        (combined_votes, f"{morph}.fits"),
                        (combined_n, "n.fits"),
                    ]

                    for f, n in to_write:
                        fits.PrimaryHDU(data=f).writeto(
                            os.path.join(out_dir, n), overwrite=True
                        )

    @staticmethod
    def _merge_parallel_means_vars(
        total_mean: np.ndarray,
        total_var: np.ndarray,
        total_n: np.ndarray,
        new_mean: np.ndarray,
        new_var: np.ndarray,
        new_n: np.ndarray,
        y_idx: int,
    ) -> None:
        """Merge merge means/vars from a new piece to total.

        Derived from:
        https://www.emathzone.com/tutorials/basic-statistics/combined-variance.html

        Args:
            total (np.ndarray): The array of means to add ``new`` to
            total_n (np.ndarray): The array of counts to add ``new_n`` to
            new (np.ndarray): the new means to add to ``total``
            new_n (np.ndarray): the new counts to add to ``total``
            y_idx (int): index for placement of ``new`` into ``total`` along y axis

        Returns:
            None
        """
        ys = slice(y_idx, y_idx + new_mean.shape[0])

        x1, x2 = total_mean[ys, :].copy(), new_mean.copy()
        s1, s2 = total_var[ys, :].copy(), new_var.copy()
        n1, n2 = total_n[ys, :].copy(), new_n.copy()

        denominator = n1 + n2

        xc_numerator = n1 * x1 + n2 * x2
        xc = np.where(denominator > 0, xc_numerator / denominator, 0)

        sc_numerator = (n1 * (s1 + np.square(x1 - xc))) + (
            n2 * (s2 + np.square(x2 - xc))
        )
        sc = np.where(denominator > 0, sc_numerator / denominator, 0)

        total_mean[ys, :] = xc
        total_var[ys, :] = sc
        total_n[ys, :] = denominator

    @staticmethod
    def _merge_parallel_rank_votes(
        total_votes: np.ndarray,
        total_n: np.ndarray,
        new_votes: np.ndarray,
        new_n: np.ndarray,
        y_idx: int,
    ) -> None:
        """Merge vote counts from a new piece to total

        Args:
            total_count (np.ndarray): The array of votes to add ``new`` to
            total_n (np.ndarray): The array of counts to add ``new_n`` to
            new_votes (np.ndarray): The array of votes to add to ``total``
            new_n (np.ndarray): The array of counts to add to ``new``
            y_idx (int): index for placement pf ``new`` into ``total`` along y axis

        Returns:
            None
        """
        ys = slice(y_idx, y_idx + new_votes.shape[0])

        x1, x2 = total_votes[ys, :].copy(), new_votes.copy()
        n1, n2 = total_n[ys, :].copy(), new_n.copy()

        numerator = (n1 * x1) + (n2 * x2)
        denominator = n1 + n2
        mean = np.where(denominator > 0, numerator / denominator, 0)

        total_votes[ys, :] = mean
        total_n[ys, :] = denominator

    # TODO: Add an informative output.
    @staticmethod
    def _run_parallel_jobs(
        workers: List[int], is_gpu: bool, out_dir: str, parallel_check_interval: float
    ) -> None:
        """Starts and tracks parallel job runs.

        WARNING: This will not finish running until all subprocesses are complete

        Args:
            workers (List[int]): A list of worker ID's to assign to a portion of an
                                 image.
            is_gpu (bool): if True the worker ID's belong to NVIDIA GPUs and will
                           be used as an argument in CUDA_VISIBLE_DEVICES. If False,
                           then the ID's are assocaited with CPU workers
            out_dir (str): the location with the partitioned data
            parallel_check_interval (float): If gpus are given, then this is the number
                                             of minutes to wait between polling each
                                             subprocess for completetion

        Returns:
            None
        """

        processes = {}

        for worker in workers:
            if is_gpu:
                cmd_string = f"CUDA_VISIBLE_DEVICES={worker} python main.py"
            else:
                cmd_string = f"CUDA_VISIBLE_DEVICES=-1 python main.py"

            sub_dir = os.path.join(out_dir, str(worker))
            processes[worker] = Popen(cmd_string, shell=True, cwd=sub_dir)

        is_running = np.ones([len(workers)], dtype=np.bool)
        while is_running.any():
            for i, g in enumerate(sorted(workers)):
                if is_running[i] and (processes[g].poll() is not None):
                    is_running[i] = False

            if is_running.any():
                time.sleep(parallel_check_interval * 60)
            else:  # we're done we can skip sleep
                break

    @staticmethod
    def _validate_parallel_params(
        gpus: List[int] = None, cpus: int = None
    ) -> Tuple[List[int], bool]:
        """Validates that the parallelism scheme.

        Only one of the arguments should be given.

        Args:
            gpus (List[int]): A list of the CUDA gpu ID's to use for a
                              parallel classification.
            cpus (int): Number of cpus to use foa a parallel classification

        Returns:
            A tuple containing the list of worker ids and a boolean indicating
            wheter or not the ids belong to GPUS

        Raises:
            ValueError if both cpus and gpus are not None
        """

        # invalid params
        if (gpus is not None) and (cpus is not None):
            raise ValueError("Please only give a value cpus or gpus, not both.")

        # Simple serial run
        if (gpus is None) and (cpus is None):
            return [0], False

        if gpus is not None:
            if len(gpus) == 1:
                err = "Only one gpus indicated. If you are trying to select "
                err += "a single gpu, then use the CUDA_VISIBLE_DEVICES environment "
                err += "variable. For more information visit: "
                err += "https://devblogs.nvidia.com/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/"

                raise ValueError(err)
            else:
                return gpus, True
        else:
            if cpus < 2:
                raise ValueError(
                    "If passing cpus please indicate a value greater than 1."
                )

            return np.arange(cpus), False

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
        """Aggregates pixel level morphological classifications to the source level.

        Uses a flux-weighted mean of the pixel level morphologies to calculate
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
