*******************************
dataset_utils.py
*******************************

Summary
------------
**Path:** :file:`/data/groups/comp-astro/morpheus/dataset_utils.py`

This file hosts a few helper functions not included within the morpheus class that are used in other files. This includes a segmentation map creating function for the five morphological classifications, a function to isolate the target galaxy in an image, a function to create stamps, another function to mask out galaxy companions, and a function to plot segmentation maps. 

Functions
------------
Functions used in this file:

.. py:function:: mask_galaxy(img, segmap, segval, n_iter: int = 3, shuffle: bool = False)
    
    Create and return a mask for the inputted three dimensional image. Locally adds a third dimension to segmap and creates a binary mask for segmented sources. Creates a background mask by performing the logical not on the segmented source binary mask. 
    Creates a binary mask for the central galaxy by parsing through the segmap (whose shape is presumably (256,256,3)) and assigning 1 to the output image when pixel equals segval and 0 otherwise. Creates a random background whose values range from 1 with a standard deviation of the background mask. Should be updating input image and updating all pixels that aren't of the target galaxy with the randomly generated background mask, but appears to be doing the opposite...

    :param img: Input image. Dimensions should be (x,y,colors).
    :type img: numpy.ndarray
    :param segmap: Input segmentation map. Dimensions should be (x,y).
    :type segmap: numpy.ndarray
    :param segval: The value used to identify the central galaxy in the segmap. Scalar int. Must be a value in segmap's range of values. 
    :type segval: int
    :param n_iter: Optional number of iterations for binary dilation (expansion of an image by a pattern). Default is 3.
    :type n_iter: int or None
    :param shuffle: Not used by this function. Optional flag indicating whether to shuffle the pixels. Default is False.  
    :type shuffle: boolean or None
    :return: Mask of input image with nontarget galaxies essentially hidden. Dimensions are the size of the original image: (x,y,colors).
    :rtype: numpy.ndarray

.. py:function:: make_proba_seg(rgb_img, jades_seg, jades_cat, matched_cat, jades_xs, jades_ys, valid_indices, n_gals= None, size=300, additional_class=False, mag_lim=None, n_mophos=5)

    Create segmentation maps for RGB input image.

    :param rgb_img: Input image.
    :type rgb_img: numpy.ndarray
    :param jades_seg: 
    :type jades_seg: numpy.ndarray
    :param jades_cat: Input image.
    :type jades_cat: numpy.ndarray
    :param matched_cat: Input image.
    :type matched_cat: numpy.ndarray
    :param jades_xs: X coordinates.
    :type jades_xs: numpy.ndarray
    :param jades_ys: Y coordinates.
    :type jades_ys: numpy.ndarray

    :return: Mask applied to input image.
    :rtype: numpy.ndarray

.. py:function:: make_stamps_per_classes(field_name, rgb_img, wcs, jades_seg, cat_jades, n_gals=None, n_plots=50, size=100, clean=True, proba_cut=False, keep_bckg_second_guess=False)
    
    Creates stamps for each of the classes.

    :param field_name: Field name for plot title.
    :type field_name: str
    :param rgb_img: Input image.
    :type rgb_img: numpy.ndarray
    :param wcs: WCS coordinates.
    :type wcs: astropy.wcs 
    :param jades_seg: Segmentation map.
    :type jades_seg: numpy.ndarray
    :param cat_jades: Input image.
    :type cat_jades: numpy.ndarray
    :param n_gals: Number of galaxies.
    :type n_gals: int
    :param n_plots: Number of plots.
    :type n_plots: int
    :param size: Input image size.
    :type size: integer
    :param clean: Masks out companions if set, likely cleans image quality.
    :type clean: bool
    :param proba_cut: Not used within function.
    :type proba_cut: bool
    :param keep_bckg_second_guess: If set, will not append flux stamps if very sure they are background.
    :type keep_bckg_second_guess: bool

    :return: List of flux stamps categorized in four classes: spheroid, disk, irregular, and compact.
    :rtype: list

.. py:function:: mask_out_companions(img, segmap, segval, n_iter: int =5, shuffle: bool=False, noise_factor: int=1, noise=True)
    
    Replace central galaxy neighbours with background noise. Replace the detected sources around the central galaxy with either randomly selected pixels from the background or a random realisation of the background noise.

    :param img: The input image represented as a NumPy array.
    :type img: numpy.ndarray
    :param segmap: The segmentation map represented as a NumPy array.
    :type segmap: numpy.ndarray
    :param segval: The value used to identify the central galaxy in the segmap.
    :type segval: int
    :param n_iter: The number of iterations for binary dilation. Default is 5.
    :type n_iter: int or optional
    :param shuffle: A flag indicating whether to shuffle the pixels. Default is False.
    :type shuffle: bool
    :param noise_factor: A factor to control the noise intensity. Default is 1.
    :type noise_factor: int
    :param noise: A flag indicating whether to apply noise to the masked image. Default is True.
    :type noise: bool or optional

    :return: A NumPy array representing the masked image, where the central galaxy neighbors have been replaced with background noise or random background pixels.
    :rtype: numpy.ndarray

.. py:function:: plot_segs(img, img_copy, xs, xe, ys, ye, seg, name)
    Plots flux image and segmentation maps by class.
    :param img: Flux image.
    :type img: numpy.ndarray
    :param xs: Start x index of segmentation map.
    :type xs: int
    :param xe: End x index of segmentation map.
    :type xe: int
    :param ys: Start y index of segmentation map.
    :type ys: int    
    :param ye: End y index of segmentation map.
    :type ye: int
    :param seg: Segmentation map to be plotted.
    :type seg: numpy.ndarray
    :param name: Name of image for plot to be saved as.
    :type name: str

