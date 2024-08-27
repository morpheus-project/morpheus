# Sierra Janson
# MIT License
# 05/23/2024 - Present

import os
import argparse
import pyvo
import time
import requests
from pyvo.dal.adhoc import DatalinkResults, SodaQuery
from astropy.io import fits
from astropy import nddata
import numpy as np
import matplotlib.pyplot as plt    

BUFFER = 1
FLUX_IMG_SHAPE = (4100,4200)     

DATA_PATH = "data" 
DATA_PATH_RAW = os.path.join(DATA_PATH, "raw")


#######################################
# Create command line argument parser
#######################################

def create_parser():
    """Specifies arguments for program, including:
        --filename (str): Desired filename for downloaded LSST flux image | default="image{buffer}_{ra}_{dec}.fits",
        --buffer (int): Numeric identifier to be included in the filenames of all images written to disk by this script, 
        --filter (str): Desired u, g, r, i, z, or y filter/band of light for LSST flux image | default="i", 
        --ra (float): Desired RA for LSST flux image,
        --dec (float): Desired DEC for LSST flux image,
        --verbose (boolean): Not used; would indicate whether informative information should be outputted,
        --tokenfilepath (str): Path that contains Rubin API secret token
        --onlyretrieveimages (boolean): True if user only desired for LSST images to be downloaded; False for entire script to run,
        --lsstimagespath (str): Path that LSST images are downloaded to | default="data/raw/",
        --outputpath (str): Path that this script's output files are written to | default="data/raw/"
    
    Args:
        None
    
    Returns:
        Parser object.
    """
    # Handle user input with argparse
    parser      = argparse.ArgumentParser(
    description = "Detection flags and options from user.")
    
    # input file name (with ra, dec, filter) argument
    parser.add_argument(
        '-n', 
        '--filename',
        dest='filename',
        type=str,
        default=None,
        metavar='filter',
        required=False
    
    )
    parser.add_argument(
	'-b',
	'--buffer',
	dest='buffer',
	type=int,
	required=False
    )
    
    # specify filter argument
    parser.add_argument(
        '-f', '--filter',
        dest='filter',
        default="i",
        metavar='filter',
        required=False,
        help='Band of desired image.'
    )

    # specify ra argument
    parser.add_argument('-r', '--ra',
        dest='ra',
        default=62,
        type=float,
        required=False,
        help="Ra of desired image")

    # specify dec argument
    parser.add_argument('-d', '--dec',
        dest='dec',
        default=-37,
        type=float,
        required=False,
        help="Dec of desired image")

    # specify whether helpful information should be outputted
    parser.add_argument('-v', '--verbose',
        dest='verbose',
        action='store_true',
        help='Print helpful information to the screen? (default: False)',
        default=False)
    
    # specify path to file with Rubin API token
    parser.add_argument('-t', '--tokenfilepath',
        dest='tokenfilepath',
        type=str,
        required=False,
        help='Provide path to file with Rubin API token',
        default=None)
    
    # LSST image-retrieval only
    parser.add_argument('--onlyretrieveimages',
        required=False,
        action='store_true',
        help='Specify that user only wants to query and retrieve LSST images, not perform sersic fitting.',
        default=False)
    
    # specify path to write LSST images to
    parser.add_argument('--lsstimagespath',
        dest='lsstimagespath',
        type=str,
        required=False,
        help='Specify that user only wants to query and retrieve LSST images, not perform sersic fitting.',
        default=DATA_PATH_RAW)
    
    # specify path to write Sersic outputs of script (including retrieved LSST image) to 
    parser.add_argument('--outputpath',
        dest='outputpath',
        type=str,
        required=False,
        help='Specify path that segmentation map (.npy), Sersic fit values & PSF FITS table, and class-labelled segmentation FITS image are written to.',
        default=DATA_PATH_RAW)

    return parser

#######################################
# authenticate() function
#######################################
def authenticate(tokenfilepath):
    """Follow linked. instructions from RSP below to set up the API and retrieve your own token. ENSURE you keep your token private. https://dp0-2.lsst.io/data-access-analysis-tools/api-intro.html
    
    Args:
        tokenfilepath (str): Path that LSST images are downloaded to
        
    Returns: 
        Authenticated service object if successful.
        
    Raises:
        FileNotFoundError if tokenfilepath is not recognized
        Exception if any error occurs
    """
    RSP_TAP_SERVICE = 'https://data.lsst.cloud/api/tap'
    token_file      = ''

    # if a file path was provided
    try:
        if (tokenfilepath != None): 
            token_file         = tokenfilepath
            
        else:
            homedir            = os.path.expanduser('~')
            secret_file_name   = ".rsp-tap.token" 
            token_file         = os.path.join(homedir,secret_file_name)
            
        with open(token_file, 'r') as f:
            token_str = f.readline()
        
        cred       = pyvo.auth.CredentialStore()
        cred.set_password("x-oauth-basic", token_str)
        credential = cred.get("ivo://ivoa.net/sso#BasicAA")
        service    = pyvo.dal.TAPService(RSP_TAP_SERVICE, credential)
        return service
    
    except FileNotFoundError:
        raise Exception("The path you provided is not recognized by the system.")
        
    except:
        raise Exception("Something failed while attempting to set up the API. This may be because you are not using a Linux system. In which case you will have to alter the 'authenticate()' function in the code yourself, until I adjust this.")
    
    
#######################################
# main() function
#######################################
def image_retrieval(args):
    """Retrieves LSST flux images from the Rubin Science Platform.
    
    Args:
        args (argparse Object): Arguments provided at commandline.
    
    Returns (List): List of paths of images downloaded. 
    """
    import json
    import datetime
    
    
    # begin timer
    time_global_start = time.time()

    # authenticate TAPS
    service = authenticate(args.tokenfilepath)

    ra     = []
    dec    = []
    filter = []

    iterations = 0
    if (args.buffer != None):
        global BUFFER
        BUFFER = args.buffer
    if (args.filename != None):
            try:
                print("Opening provided file...")
                with open(args.filename, 'r') as file:
                    data = json.load(file)
                    input_list = data["values"]
                    for element in input_list:
                        ra.append(float(element["ra"]))
                        dec.append(float(element["dec"]))
                        filter.append(element["filter"])
                        iterations+=1
            except FileNotFoundError:
                print(f"Error: The file '{args.filename}' does not exist.")
            except Exception as e:
                print(f"An error occurred: {e}")
    else:
        ra.append(args.ra)
        dec.append(args.dec)
        filter.append(args.filter)
        iterations = 1
    
    successful_images = []
    print(f"Will do {iterations} image retrievals.")
    for itera in range(iterations):
        if (args.verbose):
                global is_verbose 
                is_verbose  = True
                print("Retrieving an image with %s filter at (%f,%f)"%(filter[itera],ra[itera],dec[itera]))

        # ensuring filter is valid
        if filter[itera].lower() not in "ugrizy":
            print("The DP0.2 simulation (of whose images this script retrieves) provides images in the u, g, r, i, z, and y bands. This query will be skipped over.")
            print("Please trying again and specify one of the aforementioned filters instead.")
            continue
        
        # configuring query
        # search part
        query = """SELECT TOP %d dataproduct_type,dataproduct_subtype,calib_level,lsst_band,em_min,em_max,lsst_tract,lsst_patch, lsst_filter,lsst_visit,lsst_detector,lsst_ccdvisitid,t_exptime,t_min,t_max,s_ra,s_dec,s_fov, obs_id,obs_collection,o_ucd,facility_name,instrument_name,obs_title,s_region,access_url, access_format FROM ivoa.ObsCore WHERE calib_level = 3 AND dataproduct_type = 'image' AND lsst_band='%s' AND dataproduct_subtype = 'lsst.deepCoadd_calexp'AND CONTAINS(POINT('ICRS', %f, %f), s_region)=1"""%(1, filter[itera].lower(), ra[itera], dec[itera])

        
        # downloading images
        results = service.search(query)

        if (len(results) == 0):
                print(f'No results for query of "RA={ra[itera]}, DEC={dec[itera]}, FILTER={filter[itera]}".')
                print("The DP0.2 simulation (of whose images this script retrieves) covers 300 square degrees centered (RA, DEC) = 61.863, -35.790 degrees. Ensure your RA and DEC are within this boundary.")
        else:
                fits_images = []
                for i in range(len(results)):
                    dataLinkUrl = results[i].getdataurl()
                    auth_session = service._session
                    dl = DatalinkResults.from_result_url(dataLinkUrl, session=auth_session)
                    fits_image_url = dl.__getitem__("access_url")[0]
                    fits_images.append(fits_image_url)
                if (len(fits_images) == 0):
                        if(args.verbose):
                                print("No images retrieved. Possibly error during retrieval")
                else:
                        # retrieve & download images
                        for i in range(len(fits_images)):
                                response = requests.get(fits_images[i])
                        retrieved_imgs = 0
                        
                        # checking validity of directory
                        path = args.lsstimagespath
                        if (not os.path.exists(path)):
                                os.mkdir(path)
                                print(f'Path:"{path}" did not exist, but it does now.')
                                
                        if response.status_code == 200:
                                image_name = f"image{i+BUFFER}_{filter[itera]}_{float(ra[itera])}_{float(dec[itera])}.fits"
                                image_path = os.path.join(path,image_name)
                                with open(image_path, 'wb') as file:
                                        file.write(response.content)
                                        successful_images.append(image_path)
                                retrieved_imgs += 1
                        else:
                                if (args.verbose): print(f"Failed to download file. Status code: {response.status_code}")
                        if (args.verbose): print("Successfully downloaded %d images."%retrieved_imgs)
                        
    if (len(successful_images) == 0):
        if (args.verbose):
            print("No LSST images were successfully retrieved. The script cannot continue, and will exit here.")
        quit()
    #end timer
    time_global_end = time.time()
    if(args.verbose):
            print("Time to retrieve images: %d seconds."%(time_global_end-time_global_start))
    
    with open(f"downloaded_LSST_images.txt", "a") as file:
        img_catalog = open("downloaded_LSST_images.txt","r")
        downloaded_images = img_catalog.readlines()
        
        imagecount = len(downloaded_images)
        for i in successful_images:
            file.write(f"{imagecount}")
            file.write("\t")
            file.write(f"{datetime.datetime.now()}")
            file.write("\t")
            file.write(i)
            file.write("\n")
            imagecount+=1
    
    return successful_images

def set_up(image_path):
        """Returns image, variance, and a graphable PSF from a provided FITS filepath
        
        Args:
            image_path (str): Path to LSST flux image
        
        Returns:
            (List[numpy.ndarray, numpy.ndarray, numpy.ndarray, astropy.io.fits.hdu.hdulist.HDUList]): Flux image data, variance image data, PSF image, and template HDU for output
            
        """
        hdul            = fits.open(image_path)

        image           = hdul[1].data            # image
        variance        = hdul[3].data            # variance
        
        # initializing image shape
        global FLUX_IMG_SHAPE
        FLUX_IMG_SHAPE = image.shape
        
        psfex_info      = hdul[9]
        psfex_data      = hdul[10]
        pixstep         = psfex_info.data._pixstep[0]  # Image pixel per PSF pixel
        size            = psfex_data.data["_size"]  # size of PSF  (nx, ny, n_basis_vectors)
        comp            = psfex_data.data["_comp"]  # PSF basis components
        coeff           = psfex_data.data["coeff"]  # Coefficients modifying each basis vector
        psf_basis_image = comp[0].reshape(*size[0][::-1])
        psf_image       = psf_basis_image * psfex_data.data["basis"][0, :, np.newaxis, np.newaxis]
        psf_image       = psf_image.sum(0)
        psf_image /= psf_image.sum() * pixstep**2
        
        # open template_hdu for writing
        argstemplate = 'morph_cat_template.fits'
        template_hdu = fits.open(argstemplate)
        
        # Appending PSF to FITS output
        image_hdu2 = fits.ImageHDU(data=psf_image, name="PSF")
        template_hdu.append(image_hdu2)
        
        return [image, variance, psf_image, template_hdu]

def identify_sources(variance, image, threshold=25):
        """Identify sources in fits image by threshold of how bright they are with respect to the median of the variance image
        
        Args:
            variance (numpy.ndarray): Variance image data
            image (numpy.ndarray): Flux image data
            threshold (int): Multiplier of median of variance image to determine threshold for an object to be considered a source by pysersic's detect_sources function
        
        Returns:
            Segmentation map returned by Pysersic
        
        
        """
        # from photutils.background   import Background2D, MedianBackground
        from astropy.convolution    import convolve
        from photutils.segmentation import detect_sources
        from photutils.segmentation import make_2dgaussian_kernel # not comptabile with py 3.7
        from photutils.segmentation import deblend_sources
        from skimage.morphology import binary_opening

        # deblend sources to prevent overlapping sources in cutouts
        
        bkg              = np.median(variance)       # perhaps change to rms of bkg
        threshold        = threshold * bkg 

        segment_map      = detect_sources(image, threshold, npixels=10)
        
        # perform erosion and dialation
        # img_changed      = binary_opening(image)
        # deblended        = deblend_sources(image, segment_map,10, contrast=0.45)
        # segment_map.remove_border_labels(5, partial_overlap=False, relabel=True)
        return segment_map


# HELPER FUNCTIONS ----------------------#
def resize_image(psf_image, new_shape):
    """Resizes PSF (or any passed image) to image size as per specfication of pysersic's FitSingle using cv2 interpolation.
    
    Args:
        psf_image (numpy.ndarray): PSF image
        new_shape (Tuple[int, int]): Square size that image will be changed to
    
    Returns:
        Resized PSF
    """
    import cv2
    resized_psf = cv2.resize(psf_image, new_shape, interpolation=cv2.INTER_AREA)
    resized_psf /= np.sum(resized_psf)    
    return resized_psf    

def smooth_seg(segment_map):
    """Expands boundaries around all sources by 1 pixels
    
    Args:
        segment_map (photutils.segmentation.core.SegmentationImage): Segmentation map of LSST flux image returned from pysersic's detect_sources function
        
    Returns
        (numpy.ndarray): Segmentation map with the radius of all sources increased by one pixel to smooth out image appearence
        
    """
    from skimage.segmentation import expand_labels
    segmap = np.array(segment_map.data)
    return expand_labels(segmap,distance=1)

    
def create_cutouts(segment_map, image, variance, psf):
        """Create same dimension cutouts around sources in the image & the variance & PSF image
        
        Args:
            segment_map (photutils.segmentation.core.SegmentationImage): Segmentation map of LSST flux image returned from pysersic's detect_sources function
            image (numpy.ndarray): Data from LSST flux image
            variance (numpy.ndarray): Data from variance image
            psf (numpy.ndarray): PSF image
        
        Returns:
            cutouts (List[List[astropy.nddata.utils.Cutout2D, numpy.ndarray, jaxlib.xla_extension.ArrayImpl,  numpy.ndarray, numpy.ndarray, numpy.int64]]): List of a list of cutouts of sources of the flux image, flux image data, cutout mask data, cutout var data, PSF, and the source label from pysersic's detect_sources
        
        """
        import jax.numpy as jnp

        # grab bounding boxes (slices of cutouts)
        bbox = segment_map.bbox
        # grab source ids
        labels = segment_map.labels

        segmap_data = np.array(segment_map.data) #smooth_seg(segment_map) --> i bet it's from this
        
        # initialize mask image
        mask_img = np.logical_not(segmap_data)

        # the number of cutouts and the number of labels should always be the same
        assert(len(bbox) == len(labels))
        
        if(is_verbose): print("Creating cutouts...")
        cutouts = []
        import random
        # for each cutout from detect_sources, cutout a mask image, variance image, and psf image
        for i in range(len(bbox)):
                y_center, x_center = bbox[i].center
                x_len,y_len = bbox[i].shape
                min_length = 12 #22
                
                # if (x_len> 10 and y_len > 10 and x_len < 40 and y_len < 40):
                length        = max([x_len, y_len, min_length]) * 1.25
                cutout_img    = nddata.Cutout2D(image, (x_center,y_center), int(length))

                cutout_length = max(cutout_img.shape[0], cutout_img.shape[1])
                cutout_shape  = (cutout_length,cutout_length)

                # make mask out of segmentation map cutout
                xs, ys        = cutout_img.slices_original
                xmin, xmax    = xs.start, xs.stop
                ymin, ymax    = ys.start, ys.stop

                # if the image is not a square
                if (cutout_img.data.shape[0] != cutout_img.data.shape[1]):

                    def recrop(image, x_len, y_len, length):
                        lchange = length//2
                        rchange = length//2 + length%2
                        if (x_len < y_len): # x < y
                            if (xmin - lchange < 0 or xmax + rchange >= FLUX_IMG_SHAPE[1]):
                                difference = y_len - x_len
                                lchange = difference//2
                                rchange = difference//2 + difference%2
                                return image[xmin:xmax, ymin+lchange:ymax-rchange]

                            return image[xmin-lchange: xmax+rchange, ymin:ymax]

                        else: # dimension == "y" <--> y < x
                            if (ymin - lchange < 0 or ymax + rchange >= FLUX_IMG_SHAPE[1]):
                                difference = x_len - y_len
                                lchange = difference//2
                                rchange = difference//2 + difference%2
                                return image[xmin+lchange:xmax-rchange, ymin:ymax]

                            return image[xmin:xmax, ymin-lchange: ymax+rchange]

                    x_len = cutout_img.data.shape[0]
                    y_len = cutout_img.data.shape[1]
                    img_data              = recrop(image, x_len, y_len, cutout_length) 
                    resized_var           = recrop(variance, x_len, y_len, cutout_length)
                    resized_seg_cutout    = recrop(segmap_data, x_len, y_len, cutout_length)
                    resized_mask          = jnp.array(resized_seg_cutout != labels[i])  # if segment_map == source_id = False (0) else True (1)
                    actual_psf            = resize_image(psf, img_data.shape)
                    assert(resized_mask.shape == img_data.shape)
                    package               = [cutout_img, img_data, resized_mask, resized_var, actual_psf, labels[i]] 
                    cutouts.append(package)
                else: # cutout is square
                    seg_cutout    = segmap_data[xmin:xmax,ymin:ymax]
                    cutout_mask   = jnp.array(seg_cutout != labels[i])  # if segment_map == source_id = False (0) else True (1)
                    actual_psf    = resize_image(psf, cutout_shape) # before not reversed
                    cutout_var    = nddata.Cutout2D(variance, (x_center,y_center), int(length))
                    package       = [cutout_img,cutout_img.data, cutout_mask, cutout_var.data, actual_psf,labels[i]] 
                    cutouts.append(package)
        return cutouts

def determine_class(n):
    """Determines Sérsic class based on Sérsic index
    
    Args:
        n (int): Sérsic index)
    
    Returns
        (int): Sérsic class 
    
    """
    if (0 < n and n < 1.5):
        return 1
    if (1.5 <= n and n < 3):
        return 2
    if (3 <= n and n < 5):
        return 3
    if (5 <= n):
        return 4
    return 5
    
    
def cutout_sersic_fitting(template_hdu, cutouts):
    """Fit Sérsic profiles to source cutouts using pysersic
    
    Args:
        template_hdu (astropy.io.fits.hdu.hdulist.HDUList): HDU for FITS table for Sérsic values to be written to
        cutouts (List[List[astropy.nddata.utils.Cutout2D, numpy.ndarray, jaxlib.xla_extension.ArrayImpl,  numpy.ndarray, numpy.ndarray, numpy.int64]]): From create_cutouts function; List of a list of cutouts of sources of the flux image, flux image data, cutout mask data, cutout var data, PSF, and the source label from pysersic's detect_sources 
        
    Returns:
        template_hdul written with Sérsic fit values and hdul for layered segmentation map
    
    Raises:
        AssertionError if the image cutout differs in shape from the model cutout
        Exception if there is any error with a cutout image
    
    """
    from pysersic import FitSingle
    from pysersic.priors import SourceProperties
    from pysersic import check_input_data
    from pysersic import FitSingle
    from pysersic.loss import gaussian_loss
    from pysersic.results import plot_residual
    from jax.random import PRNGKey
    import warnings
    warnings.filterwarnings("ignore")

    
    ##############################################
    # Initializing FITS file for output ---------#
    ##############################################
    # creating fits template
    idxh = {'PRIMARY':0, 'STAT_TABLE':1}
    
    n_obj = 1#len(cutouts) # number of sources
    
    primary_hdu          = template_hdu[idxh['PRIMARY']]
    stats_template       = template_hdu[idxh['STAT_TABLE']].data
    stats_cat            = fits.FITS_rec.from_columns(stats_template.columns, nrows=n_obj, fill=True)
    
    # initialize segmap FITS image
    layers = [fits.PrimaryHDU()]
    num_classes = 4

    for i in range(num_classes):
        layers.append(fits.ImageHDU(data=np.zeros(FLUX_IMG_SHAPE),name=f"Segmentation Layer {i}"))
    hdul = fits.HDUList(layers)
    layers.append(np.zeros(FLUX_IMG_SHAPE))
    
    # initialize markdown for metadata
    markdown = "segmap_info.md" 
    with open(markdown, "w") as md:
        md.write("# Segmentation Map with Sersic Index Classes Description\n\n")
        md.write("Each Image HDU is labelled with either 0, or a Sersic index class # (defined below) from (1-5). These HDUs can be stacked together to create a nuanced image.")
        md.write("\n## Sersic Index Classes:\n\n")
        md.write("Class #1. 0 < n < 1.5\n"
                 "Class #2. 1.5 < n < 3\n"
                 "Class #3. 3 < n < 5\n"
                 "Class #4. n > 5\n"
                 "Class #5. n == 0\n")
    
    
    ##############################################
    # Sersic Fitting Each Cutout ----------------#
    ##############################################

    for i in range(n_obj):
        im,im_data,mask,sig,psf,label = cutouts[i] # image, mask, variance, psf
        if (im_data.shape[0] != im_data.shape[1]):
                print('This should not happen. Pysersic dictates that the cutouts must be square.')
                print(im_data.shape)
        else:
            try:
                # Prior Estimation of Parameters
                props = SourceProperties(im_data,mask=mask) 
                prior = props.generate_prior('sersic',sky_type='none')

                fitter     = FitSingle(data=im_data,rms=sig, psf=psf, prior=prior, mask=mask, loss_func=gaussian_loss) 
                map_params = fitter.find_MAP(rkey = PRNGKey(1000));                      # contains dictionary of Sersic values

                # can see residual plot of model and flux cutout if desired: 
                # fig, ax = plot_residual(im.data,map_params['model'],mask=mask,vmin=-1,vmax=1);

                ##############################################
                # Testing Fit -------------------------------#
                ##############################################
                image   = im_data
                xc      = map_params["xc"]
                yc      = map_params["yc"]
                flux    = map_params["flux"]
                r_eff   = map_params["r_eff"]
                n       = map_params["n"]
                ellip   = map_params["ellip"]
                theta   = map_params["theta"]
                model   = map_params["model"]
                assert(image.shape == model.shape)

                n_class       = determine_class(n) 

                # retrieving original indices of cutout
                xs, ys        = im.slices_original
                xmin, xmax    = xs.start, xs.stop
                ymin, ymax    = ys.start, ys.stop

                # creating cutout of segmap & labelling with class #
                segmap_cutout = np.logical_not(mask) 
                x_coords, y_coords = np.where(segmap_cutout == 1)
                coords = zip(x_coords, y_coords) 
                for (x,y) in coords:
                        layers[n_class].data[x+xmin,y+ymin] = n_class

                # writing background class
                # bg_x_coords,bg_y_coords = np.where(segmap_cutout == 0)
                # bg_coords = zip(bg_x_coords, bg_y_coords)
                # for (x,y) in bg_coords:
                #     if (x+xmin > 4100 or y+ymin>4200): 
                #         yikes_count += 1
                #     layers[5].data[x+xmin,y+ymin] = 1


                # writing this to specific layer in FITS image 
                # layers[n_class].data[x_min:x_max, y_min:y_max] = segmap_cutout


                # Chi-squared Statistic ----------------------------------------------------------------#
                # (evaluating whether the difference in Image and Model is systematic or due to noise)
                from scipy.stats import chi2

                chi_square           = np.sum((image*2.2 - model) ** 2 / (model))
                df                   = image.size-1                                      # number of categories - 1
                p_value              = chi2.sf(chi_square, df)

                #L1-Norm 
                noise_threshold      = np.mean(sig.data)                               
                image_1D             = image.flatten()
                model_1D             = model.flatten()
                difference_1D        = image_1D - model_1D

                l1                   = np.sum(np.abs(difference_1D))
                l1_normalized        = l1/(image_1D.size)
                l1_var_difference    = l1_normalized - noise_threshold

                ##############################################
                # Creating a FITS image with relevant data --#
                ##############################################
                x,y            = im.center_cutout
                ccols          = [label,x,y] #id, x, y
                morph_params   = [xc, yc, flux, n, r_eff, ellip, theta]
                stats          = [p_value,l1_var_difference]
                values         = ccols + morph_params + stats

                for j in range(len(values)):
                    stats_cat[i][j] = values[j]

            except Exception as error:
                    print(f"error with image number {i}.")
                    print(f"Error: {error}")

    template_hdu[idxh['STAT_TABLE']].data = stats_cat

    return template_hdu, hdul
                                
def main(args, image_path, i):
        """Manager function for entire process
        
        Args:
            args (argparse object): Command line arguments provided
            image_path (str): Path to LSST flux image
            i (int): Iteration per LSST image downloaded in one script run. Most likely will be 0 unless user downloads multiple LSST images in one run of script.
        
        Returns: 
            None
        """
        time_global_start = time.time()
        
        # process:
        image, variance, psf, template_hdu    = set_up(image_path)                                            # grabbing image, variance, psf, and output file template
        segment_map                           = identify_sources(variance, image)                             # grabbing segmentation map (with source-ids)
        cutouts                               = create_cutouts(segment_map, image, variance, psf)             # create cutouts of flux image, mask, psf, variance
        template_hdu, segmap_hdul             = cutout_sersic_fitting(template_hdu, cutouts)                  # fit each cutout with sersic values and save to template_hdu
        
        path = args.outputpath
        
        if (not os.path.exists(path)):
            os.mkdir(path)
            print(f'Path:"{path}" did not exist, but it does now.')

        # writing to disk: 
        seg_fname = os.path.join(path, f'seg{i+BUFFER}.npy')
        np.save(seg_fname, segment_map)                 # writing segmentation map (.npy) to disk
        
        vals_fname = os.path.join(path, f'morph-stats{i+BUFFER}.fits')
        template_hdu.writeto(vals_fname, overwrite=True)       # writing sersic values to disk
        
        labelledseg_fname = os.path.join(path, f'labelled_segmap{i+BUFFER}.fits')
        segmap_hdul.writeto(labelledseg_fname, overwrite=True)   # writting labelled segmap to disk
        
        with fits.open(vals_fname) as hdul:                   # open and print stats table for verification
            hdul.info()
            print(hdul[1].data)
        
        time_global_end = time.time()
        if(is_verbose): print("Time to perform sersic fitting: %d minutes"% ((time_global_end-time_global_start)//60))


#######################################
# Run the program
#######################################
if __name__=="__main__":    
    
    # create the command line argument parser
    parser = create_parser()
    # store the command line arguments
    args = parser.parse_args()
    
    # obtain paths to LSST FITS images
    retrieved_images = image_retrieval(args)    
        
    if not args.onlyretrieveimages:
        
        # fit each queried image with a sersic profile
        for i in range(len(retrieved_images)):
            main(args, retrieved_images[i], i)
                
