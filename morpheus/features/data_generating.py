# Ryan Hausen
# Modifications for Rubin-specific data by Sierra
# MIT License

import os
import argparse
import random
from functools import partial
from itertools import takewhile
from typing import List, Tuple

import numpy as np
# from skimage.draw import disk
from astropy.io import fits
# from astropy.nddata.utils import Cutout2D
from astropy.wcs import WCS
# from sklearn.preprocessing import minmax_scale
# from skimage.measure import regionprops
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# directories for data to be written to 
# name data_path the name of your desired output directory
DATA_PATH = "data" 
DATA_PATH_PROCESSED = os.path.join(DATA_PATH, "processed")
DATA_PATH_PROCESSED_TRAIN = os.path.join(DATA_PATH_PROCESSED, "train")
DATA_PATH_PROCESSED_TEST = os.path.join(DATA_PATH_PROCESSED, "test")
DATA_PATH_RAW = os.path.join(DATA_PATH, "raw")

NUM_TRAIN_EXAMPLES = 1000 
NUM_TEST_EXAMPLES = 100


def make_dirs():
    """Writes directories (default are data/processed/train, data/processed/test, data/raw/train, data/raw/test) to disk if they don't already exist
    
    Args:
        None
        
    Returns:
        None
    
    """
    dirs = [
        DATA_PATH,
        DATA_PATH_PROCESSED,
        DATA_PATH_PROCESSED_TRAIN,
        DATA_PATH_PROCESSED_TEST,
        DATA_PATH_RAW,
    ]
        
    mk = lambda s: os.makedirs(s) if not os.path.exists(s) else None
    for _ in map(mk, dirs):  # apply func to each element
        pass

def validate_idx(
    mask: np.ndarray,
    upper_y: int,
    upper_x: int,
    img_size: int,
    yx: Tuple[int, int],
) -> bool:
    """Assesses if provided indices are valid for slicing the flux image and that the sliced region is valid (not masked out)
    
    Args:
        mask (np.ndarray): The mask of the flux image
        upper_y (int): The upper y boundary for which the flux image can be indexed
        upper_x (int): The upper x boundary for which the flkux image can be indexed
        img_size (int): The length or width of the desired training image (assuming square dimensions)
        yx: (Tuple[int, int]): The randomly generated slice from the idx_generator function ordered (y,x)
    
    Returns:
        True if the mask slice is consisted of all ones, false otherwise.
    
    """
    y, x = yx
    
    if y + img_size > upper_y or x + img_size > upper_x:
        return False

    slice_y = slice(y, y + img_size)
    slice_x = slice(x, x + img_size)
    
    return mask[slice_y, slice_x].copy().all()


def idx_generator(start: int, end: int) -> int:
    """Randomly generates integers between an integer start and end bound; used for random slice generation.
    
    Args:
        start (int): start bound (starting valid index) of image for random index selection
        end (int): start bound (starting valid index) of image for random index selection
    Returns:
        Randomly generated number between start and end bounds (int)
    """
    while True:
        yield random.randint(start, end)


def make_idx_collection(
    mask: np.ndarray,
    img_size: int,
    collection_size: int,
    start_y: int,
    end_y: int,
    start_x: int,
    end_x: int,
) -> List[int]:

    """Randomly generates and returns x valid slices to create training images out of in a list where x == collection_size 
    
    Args:
        mask (np.ndarray): The mask of the flux image
        img_size (int): The length or width of the desired training image (assuming square dimensions)
        collection_size (int): Number of valid slices the function will collect.
        start_y (int): Start y bound (starting valid index) of image for random index selection
        end_y (int): End y bound (starting valid index) of image for random index selection
        start_x (int): Start x bound (starting valid index) of image for random index selection
        end_x (int): End x bound (starting valid index) of image for random index selection

    Returns:
        enumerated valid indices (List[int]): An enumerated list of valid indices in the format [(i, (y, x)), ..., (collection_size-1, (y, x))]
    
    """
    y_gen = idx_generator(start_y, end_y)
    x_gen = idx_generator(start_x, end_x)

    valid_count = lambda iyx: iyx[0] < collection_size
    valid_idx_f = partial(validate_idx, mask, end_y, end_x, img_size)

    # We want a collection of valid (y, x) coordinates that we can use to
    # crop from the larger image. Explanation starting from the right:
    #
    # zip(y_gen, x_gen) -- is an inifite generator that produces random integer
    #                      tuples of (y, x) within the start and end ranges
    #
    # filter(valid_idx_f, ...) -- filters tuple pairs from zip(y_gen, x_gen)
    #                             that can't be used to generate a valid sample
    #                             based on the conditions in valid_idx_f function
    #
    # enumerate(...) -- returns a tuple (i, (y, x)) where i is an integer that
    #                   counts the values as they are generated. In our case,
    #                   its a running count of the valid tuples generated
    #
    # takewhile(valid_count, ...) -- takewhile returns the values in the
    #                                collection while valid_count returns True.
    #                                In our case, valid_count returns true while
    #                                the i returned by enumerate(...) is less
    #                                than collection_size
    #
    # [iyx for iyx ...] -- iyx is a valid tuple (i, (y, x)) from takewhile(...)
    return [
        iyx
        for iyx in takewhile(
            valid_count, enumerate(filter(valid_idx_f, zip(y_gen, x_gen)))
        )
    ]


# Adapted from https://docs.astropy.org/en/stable/nddata/utils.html#saving-a-2d-cutout-to-a-fits-file-with-an-updated-wcs
def crop_and_save(
    data: np.ndarray,
    wcs: WCS,
    img_size: int,
    save_dir: str,
    iyx: Tuple[int, Tuple[int, int]],
    buffer: int
) -> None:
    """Uses slices to crop images out of segmentation maps and write them to disk. 
    
    Args:
        data (np.ndarray): Array of flux image
        wcs (WCS): World coordinates derived from flux image header
        img_size (int): The length or width of the desired training image (assuming square dimensions)
        save_dir (str): Desired save directory
        iyx (Tuple[int, Tuple[int, int]]): List of enumerated valid indices from the make_idx_collection function
        buffer: (int) Buffer to prevent files from being overwritten
    
    Returns:
        None
    """
    i, (y, x) = iyx
    ys, xs = slice(y, y + img_size), slice(x, x + img_size)

    fits.PrimaryHDU(
        data=data[ys, xs, :].copy(), header=wcs[ys, xs].to_header()
    ).writeto(os.path.join(save_dir, f"{i+buffer}.fits"), overwrite=True)


def get_full_name(fname_key: str) -> str:
    """Helper function to retrieve the full name of a file by keyword
    
    Args:
        fname_key (str): Keyword for filename (word that should be in the file's name)
        
    Returns:
        (str): The full name of the file 
    
    """
    return next(filter(lambda f: fname_key in f, os.listdir(DATA_PATH_RAW)))


def main(img_size: int, lsst_no: List) -> None:
    """Main function: passes necessary data between functions and calls them 
    
    Args: 
        img_size (int): The length or width of the desired training image (assuming square dimensions)
        lsst_no (List[int]): Number associated with downloaded images from creating_datasets.py script (BUFFER)    
    
    Returns:
        None
    """
    
    make_dirs()

    # if (
    #     len(os.listdir(DATA_PATH_PROCESSED_TRAIN)) > 0
    #     or len(os.listdir(DATA_PATH_PROCESSED_TEST)) > 0
    # ):
    #     print(f"Files exists in {DATA_PATH_PROCESSED} skipping data extraction.")
    # else:
    random.seed(12171988)
    
    assert(lsst_no != None and "Must provide at least one buffer number (should be number on your downloaded files) as a command line option")
    lsst_img_nums = lsst_no
    
    for n in range(len(lsst_img_nums)):
        print(f"LSST image #{n+1}")
        num = lsst_img_nums[n]
        
        seg_fname = get_full_name(f"seg{num}")
        seg_path = os.path.join(DATA_PATH_RAW, seg_fname)
        segmap = np.load(seg_path).astype(np.int32)
        segmap[segmap > 0] = 1

        print("WARNING: Using a mask of all ones as a placeholder")
        mask = np.ones(segmap.shape)

        validate_fname = get_full_name(f"image{num}")
        val_path = os.path.join(DATA_PATH_RAW, validate_fname)
        val_hdul = fits.open(val_path)
        val_array = val_hdul[1].data

        
        train_ys, train_xs = (1000, segmap.shape[0]), (1000, segmap.shape[1])
        test_ys, test_xs = (0, 1000), (0, 1000)

        print('Gathering train samples...')
        train_idxs = make_idx_collection(
            mask, img_size, NUM_TRAIN_EXAMPLES, *train_ys, *train_xs 
        )

        print('Gathering test samples...')
        test_idxs = make_idx_collection(
            mask, img_size, NUM_TEST_EXAMPLES, *test_ys, *test_xs
        )
        
        del mask
        del val_array

        
        print("Opening data files")

        lbls_fname = get_full_name(f"labelled_segmap{num}")
        lbls_fpath = os.path.join(DATA_PATH_RAW, lbls_fname)
        lbls_hdul  = fits.open(lbls_fpath)

        layers = [d.data for d in tqdm(lbls_hdul[1:])]
        data = np.dstack(layers)

        print("Getting header")
        header = fits.getheader(val_path,1)
        wcs = WCS(header)

        train_crop_f = partial(
            crop_and_save, data, wcs, img_size, DATA_PATH_PROCESSED_TRAIN, buffer=NUM_TRAIN_EXAMPLES*n
        )

        for _ in map(
            train_crop_f,
            tqdm(train_idxs, desc="Making training examples", total=NUM_TRAIN_EXAMPLES),
        ):
            pass

        test_crop_f = partial(
            crop_and_save, data, wcs, img_size, DATA_PATH_PROCESSED_TEST,buffer=NUM_TEST_EXAMPLES*n
        )

        for _ in map(
            test_crop_f,
            tqdm(test_idxs, desc="Making testing examples", total=NUM_TEST_EXAMPLES),
        ):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    
    parser.add_argument("--input_size", type=int, default=800) 
    parser.add_argument("--lsst_nos", type=list_of_ints, default=None) 

    main(parser.parse_args().input_size, parser.parse_args().lsst_nos)