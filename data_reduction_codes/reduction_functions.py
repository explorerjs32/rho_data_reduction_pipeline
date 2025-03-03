import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.io.fits.verify import VerifyError
import astroalign as aa
from astroalign import MaxIterError
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_threshold, detect_sources
from photutils.utils import circular_footprint
from skimage.registration import *
from scipy.ndimage import shift
import os
import argparse
from tqdm.auto import tqdm


def get_frame_info(light_dirs, dark_dirs, flat_dirs, bias_dirs):
    """
    Extracts information from FITS file headers across multiple directories and returns it as pandas DataFrames.

    This function iterates through the provided directories, reading FITS files and extracting relevant
    header information such as object name, frame type, filter, and exposure time. It compiles this
    information into two pandas DataFrames: one containing detailed frame information and another
    summarizing the observing log with exposure counts for each object, frame type, filter, and exposure time
    combination.

    Args:
        light_dirs (list of str): List of directories containing light frames.
        dark_dirs (list of str): List of directories containing dark frames.
        flat_dirs (list of str): List of directories containing flat frames.
        bias_dirs (list of str): List of directories containing bias frames.

    Returns:
        tuple: A tuple containing the following pandas DataFrames:
            - frame_info_df: DataFrame with columns 'Directory', 'Files', 'Object', 'Frame', 'Filter', 'Exptime'
                             for each FITS file.
            - observing_log_df: DataFrame summarizing the observing log with columns 'Object', 'Frame',
                                 'Filter', 'Exptime', and 'Exposures'.
    """

    # Define the lists to store the data
    exposure_times = []
    filters = []
    frames = []
    objects = []
    file_list = []
    directories_list = []

    # Loop through all provided light frame directories
    for data_dir in light_dirs:
        # Get the list of fits files in the current directory
        current_file_list = [f for f in os.listdir(data_dir) if f.endswith('.fits') and os.path.isfile(os.path.join(data_dir, f))]

        # Loop through each file in the directory and extract header info
        for file in current_file_list:
            # Get the object name from the FITS header
            header = fits.getheader(os.path.join(data_dir, file))
            obj_name = header.get('OBJECT', 'Unknown')

            # If the object name is 'Unknown', skip this file
            if obj_name == 'Unknown':
                continue

            frame = header.get('FRAME', 'Unknown')
            exp_time = header.get('EXPTIME', 0)
            filter_ = header.get('FILTER', 'Unknown')

            # Append the information to the lists
            objects.append(obj_name)
            frames.append(frame)
            exposure_times.append(exp_time)
            filters.append(filter_)
            file_list.append(file)
            directories_list.append(data_dir)

    # Loop through all provided dark frame directories
    for data_dir in dark_dirs:
        # Get the list of fits files in the current directory
        current_file_list = [f for f in os.listdir(data_dir) if f.endswith('.fits') and os.path.isfile(os.path.join(data_dir, f))]

        # Loop through each file in the directory and extract header info
        for file in current_file_list:
            # Get the object name from the FITS header
            header = fits.getheader(os.path.join(data_dir, file))

            frame = header.get('FRAME', 'Unknown')
            exp_time = header.get('EXPTIME', 0)
            filter_ = header.get('FILTER', 'Unknown')

            # Append the information to the lists
            objects.append('Calibration')
            frames.append('Dark')
            exposure_times.append(exp_time)
            filters.append(filter_)
            file_list.append(file)
            directories_list.append(data_dir)

    # Loop through all provided flat frame directories
    for data_dir in flat_dirs:
        # Get the list of fits files in the current directory
        current_file_list = [f for f in os.listdir(data_dir) if f.endswith('.fits') and os.path.isfile(os.path.join(data_dir, f))]

        # Loop through each file in the directory and extract header info
        for file in current_file_list:
            # Get the object name from the FITS header
            header = fits.getheader(os.path.join(data_dir, file))

            frame = header.get('FRAME', 'Unknown')
            exp_time = header.get('EXPTIME', 0)
            filter_ = header.get('FILTER', 'Unknown')

            # Append the information to the lists
            objects.append('Calibration')
            frames.append('Flat')
            exposure_times.append(exp_time)
            filters.append(filter_)
            file_list.append(file)
            directories_list.append(data_dir)

    # Loop through all provided dark frame directories
    for data_dir in bias_dirs:
        # Get the list of fits files in the current directory
        current_file_list = [f for f in os.listdir(data_dir) if f.endswith('.fits') and os.path.isfile(os.path.join(data_dir, f))]

        # Loop through each file in the directory and extract header info
        for file in current_file_list:
            # Get the object name from the FITS header
            header = fits.getheader(os.path.join(data_dir, file))

            frame = header.get('FRAME', 'Unknown')
            exp_time = header.get('EXPTIME', 0)
            filter_ = header.get('FILTER', 'Unknown')

            # Append the information to the lists
            objects.append('Calibration')
            frames.append('Bias')
            exposure_times.append(exp_time)
            filters.append(filter_)
            file_list.append(file)
            directories_list.append(data_dir)

    # Generate a dataframe containing the frame information, including the directory
    frame_info_df = pd.DataFrame({
        'Directory': directories_list,
        'Files': file_list,
        'Object': objects,
        'Frame': frames,
        'Filter': filters,
        'Exptime': exposure_times
    })

    # Generate the observing log based on the frame information
    observing_log_df = frame_info_df.groupby(by=['Object', 'Frame', 'Filter', 'Exptime']).size().to_frame(name='Exposures').reset_index()

    return frame_info_df, observing_log_df

def create_master_bias(frame_info_df, log):
    '''
    Identifies bias frames, compiles and returns them using numpy median method

    Using the data extracted from fits by get_frame_info, this function compiles the various
    frames that have the bias identification into biases_files. It then proceeds to collect
    the data of each of the frames into biases_data. Finally, the collection of data is then
    combined using the numpy median method into the master_bias variable. That is then
    returned.

    Args:
        frame_info_df: (Pandas DF list) Collection of frame data at given directory
        log: List of strings containing logging to be added to the log txt file

    Returns:
        master_bias: (2D array of integers) Median of master bias data used for subsequent
        calculations

    '''

    # Filtering dataframes that are labeled as bias using df indexing
    biases_df = frame_info_df[frame_info_df["Frame"] == "Bias"].reset_index(drop=True)

    # Expanding data within dataframes of label bias into an array
    biases_data = np.array([fits.getdata(os.path.join(biases_df.loc[idx, "Directory"], biases_df.loc[idx, "Files"])).astype(float) \
                            for idx in range(biases_df["Files"].values.size)])
    
    log += [f"{len(biases_data)} bias frames were found\n"]

    # Using median combine to form a final master bias frame and then return it
    master_bias = np.median(biases_data, axis=0)

    #Using biases_data to estimate read noise
    noise=np.std(biases_data) 
    master_bias_noise=np.median(noise, axis=0)
   
    return master_bias, master_bias_noise

def create_master_darks(frame_info_df, master_bias_noise, log):
    """
    Creates a list of master darks from the information in the two dataframes.

    First, isolates the dark frames and how many unique exposures there are, then iterates through
    the list of darks for each exposure time to gather the dark frames for a specific exposure time, which
    it will median combine into a master dark.

    Args:
        frame_info_df: the frame information dataframe
        log: List of strings containing logging to be added to the log txt file

    Returns:
        dark_exposure_times: a list of master dark exposure times (float) that correlate to the master darks
        master_darks: a dictionary of master darks. Each object in the list is fits data.
            Key: "master_darks_[0.0]s" where [0.0] is replaced with the exposure time
            Value: fits data (2D array of pixel counts)
    """

    # Creating the master darks- one for each exposure time.
    # For each unique exposure (entry in observing log that is a dark frame), get that exposure time
    darks_df = frame_info_df[frame_info_df['Frame'] == 'Dark'].reset_index(drop=True)
    dark_exposure_times = darks_df['Exptime'].unique()

    # Go through the darks of that exposure length to create the master-
    master_darks = {}

    for exp in dark_exposure_times:
        darks_exp = []

        for index, row in darks_df.iterrows():
            if (row["Exptime"] == exp):
                darks_exp.append(fits.getdata(os.path.join(row['Directory'], row['Files'])))

        master_darks["master_dark_" + str(exp) + "s"] = np.median(np.array(darks_exp), axis=0)

        #Removing noise from dark frames to get the dark current
        darks_exp_array = np.array(darks_exp)
        darks_list = darks_exp_array - master_bias_noise

        #Taking median twice from the bias subtracted darks
        debiased_master_dark = np.median(darks_list, axis=0)
        dark_current= np.median(debiased_master_dark)/dark_exposure_times

        # Logging master darks created
        log += ["Master_dark_" + str(int(exp)) + "s created. " + str(len(darks_exp)) + " frames were found\n"]

        #Creating dictionary for dark current
        uncertainties_dark_current = {'dark current': dark_current}

    # return the darks and the times they correlate to.
    return dark_exposure_times, master_darks, uncertainties_dark_current

def create_master_flats(frame_info_df, darks_exptimes, master_darks, master_bias, log):
    """
     Creates a dictionary of normalized master flats for each filter from the frame information dataframe.

    The function processes flat frames by first isolating those with the 'Flat' frame type. For each unique filter,
    it retrieves the associated flat frames, corrects them using either a master dark or master bias based on
    the exposure time, and then creates a master flat by median-combining these corrected frames. Each master flat
    is normalized by dividing by its median value.

    Args:
        frame_info_df (pd.DataFrame): DataFrame containing information about the frames, including columns 'Frame',
            'Filter', and 'Files'. The 'Frame' column should have entries indicating the type of frame (e.g., 'Flat'),
            the 'Filter' column should specify the filter used, and the 'Files' column should list filenames of the frames.
        data_dir (str): Directory path where the flat frame files are located.
        darks_exptimes (list): List of exposure times for the master dark frames.
        master_darks (dict): Dictionary of master darks. Keys are strings formatted as "master_dark_[exptime]s" where
            [exptime] is the exposure time, and values are 2D numpy arrays representing the master dark frames.
        master_bias (numpy.ndarray): 2D numpy array representing the master bias frame.
        log: List of strings containing logging to be added to the log txt file

    Returns:
        tuple: A tuple containing:
            - flat_filters (list): A list of unique filter names present in the frame information dataframe.
            - master_flats (dict): A dictionary of master flats. Each key is a string "master_flat_[FilterName]" where
              [FilterName] is the filter name, and each value is a 2D numpy array representing the normalized master
              flat for that filter.
    """

    # Isolate the flat frames from the dataframe
    flats_df = frame_info_df[frame_info_df['Frame'] == 'Flat'].reset_index(drop=True)
    flat_filters = flats_df['Filter'].unique()

    # Create the master flats
    master_flats = {}

    for filter_name in flat_filters:
        flats_filter = []

        for index, row in flats_df.iterrows():
            # Get the exposure time of the flat frame
            flat_exptime = fits.getheader(os.path.join(row['Directory'], row['Files']))['EXPTIME']

            if flat_exptime in darks_exptimes and fits.getheader(os.path.join(row['Directory'], row['Files']))['FILTER'] == filter_name:
                # print(f"subtracting {flat_exptime}s master dark from {row['Files']}")
                flats_filter.append(
                    fits.getdata(os.path.join(row['Directory'], row['Files'])) - master_darks[f"master_dark_{flat_exptime}s"])

            elif flat_exptime not in darks_exptimes and fits.getheader(os.path.join(row['Directory'], row['Files']))['FILTER'] == filter_name:
                # print(f"subtracting master bias from {row['Files']}")
                flats_filter.append(fits.getdata(os.path.join(row['Directory'], row['Files'])) - master_bias)

        # Combine the flats and normalize the master flat
        master_flat = np.median(np.array(flats_filter), axis=0)
        normalized_master_flat = master_flat / np.median(master_flat)
        master_flats["master_flat_" + filter_name] = normalized_master_flat

        #Calculating uncertainty of flats
        flats_uncertainty = np.std(normalized_master_flat)

        # Logging Master flat creation
        log += ["Master_flat_" + filter_name + " created. " + str(len(flats_filter)) + " frames found\n"]

        #Creating a dictionary for the Flats uncertainty
        flats_uncertainty_dict = {'Flats uncertainty': flats_uncertainty}

    return flat_filters, master_flats, flats_uncertainty_dict

def background_subtraction(image):
    """
    Subtracts the background from an astronomical image using a 2D background estimation.

    This function estimates the sky background of the input image by assuming a non-uniform
    background brightness. The background is modeled using a sigma-clipped median estimator, 
    and sources are masked out before computing the background. The estimated background is 
    then subtracted from the input image to produce a background-subtracted image.

    Args:
        image (numpy.ndarray): 2D array representing the astronomical image from which the 
                               background will be subtracted.

    Returns:
        numpy.ndarray: The background-subtracted image.
    """

    # Define the paameters to estimate the the sky background of the image
    # A non-uniform background brightness is going to be assumed
    sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    threshold = detect_threshold(image, nsigma=2.0, sigma_clip=sigma_clip)
    segment_img = detect_sources(image, threshold, npixels=10)
    footprint = circular_footprint(radius=10)
    mask = segment_img.make_source_mask(footprint=footprint)
    box_size = (30, 30)
    filter_size = (3, 3)
    bkg_estimator = MedianBackground()
    
    # Estimate the 2D background of the image
    bkg = Background2D(image, box_size=box_size, mask=mask, filter_size=filter_size, 
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

    # Subtract the background from the image reduced image
    bkg_subtracted_image = image - bkg.background
        
    return bkg_subtracted_image

def image_reduction(frame_info_df, dark_times, master_darks, flat_filters, master_flats, master_bias, log):
    """
    Reduces raw light frame images by subtracting master darks, dividing by master flats, and applying bad pixel masks.

    This function processes light frame images by isolating them from a provided DataFrame, matching them with the 
    closest available master dark frame based on exposure time, and correcting them with the appropriate master flat 
    field for the image filter. The processed images are then background-subtracted and bad pixels are masked.

    Args:
        frame_info_df - Pandas DataFrame containing header information for every provided
        dark_times - Collection of the master_dark exposure times
        master_darks - Collection of key-value pairs of dark_frame exposure times and dark_frame data
        flat_filters - Collection of all unique filters used during the image collection
        master_flats - Dictionary of master flats
        master_bias - Master bias produced in previous operations
        data_dir - File directory
        log: List of strings containing logging to be added to the log txt file

    Returns:
        dict: Dictionary of reduced images, keyed by the original file names. Each value is a 2D numpy array representing 
              the reduced image, with the background subtracted and bad pixels masked.
    """

    # Initializes a dataframe containing all the information on raw images
    raw_image_df = frame_info_df[frame_info_df["Frame"] == "Light"].reset_index(drop=True)

    # Get the unique objects that were observed
    objects = raw_image_df['Object'].unique()

    # Define a dictionary to stored the reduced frames per object
    master_reduced_frames = {}

    # Interpolate over the different objects to reduced the data
    for object in objects:

        log += [f"Reducing raw light frames for {object}\n\n"]

        # Get a dataframe containing only the raw light frames for the given object
        object_raw_image_df = raw_image_df[raw_image_df['Object'].isin([object])].reset_index(drop=True)

        # Initialize the image reduced final product
        reduced_images = {}

        # Create the flat masks from reduced flats
        def bad_pixel(pixel_data):
            return True if (pixel_data > 2 or pixel_data < 0.5) else False

        bad_pixel_v = np.vectorize(bad_pixel)
        masks = {}
        for i in range(len(flat_filters)):
            masks["mask_" + flat_filters[i]] = bad_pixel_v(master_flats["master_flat_" + flat_filters[i]])

        # Iterate through files and create individual reduced data images
        for index in tqdm(range(len(object_raw_image_df)), desc=f"Reducing raw light frames for {object}", unit=' frames',
                          dynamic_ncols=True):

            # Identify current file, file data, and the current exposure time
            data_dir = object_raw_image_df['Directory'][index]
            file = object_raw_image_df["Files"][index]
            raw_image_data = fits.getdata(os.path.join(data_dir, file))
            raw_image_exp_time = object_raw_image_df["Exptime"][index]
            raw_image_filter = object_raw_image_df["Filter"][index]
            obj_name = fits.getheader(os.path.join(data_dir, file))['OBJECT']

            # Logging image reduction object name, file, exposure time
            log += ["Raw file: " + file + "\n"]
            log += ["Exposure time: " + str(int(raw_image_exp_time)) + " sec\n"]
            log += ["Filter: " + raw_image_filter + "\n"]

            # Identify dark_frame match OR closest match
            dark_frame = []

            # If identical match found use that given current dark
            if "master_dark_" + str(raw_image_exp_time) + "s" in master_darks:
                dark_frame = master_darks["master_dark_" + str(raw_image_exp_time) + "s"]
                log += ["Subtracted Master_dark_" + str(raw_image_exp_time) + "s\n"]


            # If not, find the closest match by:
            else:
                # Subtracting all the dark exposure times by the raw_image_exposure_time (No negative times allowed)
                temp_times = []

                for dark in dark_times:
                    temp_times.append(abs(dark - raw_image_exp_time))

                # Sorting the array to find the smallest difference
                temp_times.sort()

                # Find the closest dark_exposure_time match using implementation from
                # https://www.geeksforgeeks.org/python-find-closest-number-to-k-in-given-list/
                # Then isolate the dark frame based on the exposure time for future use
                dark_frame = master_darks["master_dark_" + str(dark_times[min(range(len(dark_times)), key=lambda i: abs(dark_times[i]-temp_times[0]))]) + "s"]

                # Logging master dark subtraction
                log += ["Subtracted Master_dark_" + str(dark_times[min(range(len(dark_times)), key=lambda i: abs(dark_times[i]-temp_times[0]))]) + "s\n"]

            # Identify flat_frame match OR provide feedback on missing filters for master_flats
            flat_frame = []
            flat_frame_found = False

            # FIXME - Fix filter issue
            if "master_flat_" + object_raw_image_df["Filter"][index] in master_flats:
                # If filter key found, use the given flat_frame
                flat_frame = master_flats["master_flat_" + object_raw_image_df["Filter"][index]]
                flat_frame_found = True

                # Logging master flat division

            else:
                # Otherwise, identify the missing filters
                print("Filter Error: missing filter " + object_raw_image_df["Filter"][index] + " for file " + object_raw_image_df["Files"][index] +
                    ". Frame type = " + object_raw_image_df["Frame"][index])

            # Perform image reduction now that everything is in place (if statement required for missing filter errors)
            # Reduce the light frames assuming no corresponding flat frame is found
            if raw_image_filter not in flat_filters:
                reduced_image = raw_image_data - dark_frame
                log += ["Flat Division skipped. No Master flat for filter: " + raw_image_df["Filter"][index] + "\n"]

            # Reduced the light frames assuming all corresponding calibration frames were found
            else:
                reduced_image = (raw_image_data - dark_frame) / flat_frame

                # log += ["Subtracted Master_dark_" + str(raw_image_exp_time) + "s\n"]
                log += ["Divided normalized_master_flat_" + raw_image_df["Filter"][index] + "\n"]

            # Subtract the background of the reduced image
            bkg_subtracted_reduced_image = background_subtraction(reduced_image)
            log += ["Subtracted background\n"]
            # Mask the bad pixels (MBP) in the background subtracted reduced science images
            if "mask_" + raw_image_filter not in masks:
                log += [f"Bad pixel mask for filter {raw_image_filter} was not found. Skipping masking\n"]

            else:
                np.putmask(bkg_subtracted_reduced_image, masks["mask_" + raw_image_filter], -999)
                final_reduced_image = bkg_subtracted_reduced_image
                log += ["Removed bad pixels\n"]

            # Store the final reduced image into the list
            reduced_images[file] = final_reduced_image        
            
            log += [file + " reduced!\n\n"]

        master_reduced_frames[object] = reduced_images

    # Finally, return the image reduced product
    return master_reduced_frames

def align_images(master_reduced_data, log):
    """
    Aligns a series of reduced astronomical images to a template image using phase cross-correlation.

    This function takes a dictionary of reduced images, identifies the brightest pixel in the first image (used as a template),
    and then aligns the remaining images to this template. Alignment is performed by determining the shift needed to match
    the template image, using phase cross-correlation, and then applying this shift to each image.

    Args:
        master_reduced_data (dict): Dictionary containing image data, where the keys are file names and the values are 2D numpy arrays 
                             representing the reduced images.
        log: List of strings containing logging to be added to the log txt file

    Returns:
        dict: A dictionary of aligned images, where the keys are the original file names and the values are the aligned 2D numpy 
              arrays. The first image in the list is returned unaltered, as it serves as the template for alignment.
    """

    # Get the object names of the reduced data
    objects = list(master_reduced_data.keys())

    # Interpolate over the objects to align the data
    master_aligned_images = {}

    for object in objects:
        log += [f"Aligning images for {object}\n\n"]

        # Get the reduced images per object
        reduced_data = master_reduced_data[object]

        # Split the image data and the file names into lists
        images = list(reduced_data.values())
        files = list(reduced_data.keys())

        # Define the template image to allign the rest to
        # This template image is always the first image of the input list
        template_image = images[0]

        # Define a list to store the aligned images
        aligned_images = {}

        # Define a list to store the file names with alignment issues
        bad_aligned_images = []

        # Interpolate over the list of images to align them
        for i, image in tqdm(enumerate(images), desc=f"Aligning images for {object}", unit=' frames',
                            total=len(images), dynamic_ncols=True):

            # If the image is the first one and the template image, add it to the list
            if i == 0:
                aligned_images[files[i]] = image

            # Else, align the rest of the images to the template
            elif i > 0:
                try:
                    # Find the transformation between the images
                    transf, (source_list, target_list) = aa.find_transform(image, template_image, max_control_points=10)
                    
                    # Apply the transformation to align image2 with image1
                    aligned_image, footprint = aa.apply_transform(transf, source=image, target=template_image)
                    aligned_images[files[i]] = aligned_image

                except(MaxIterError):
                    aligned_images[files[i]] = image
                    bad_aligned_images.append(files[i])
                    pass
        
        # Add to the log the images with alignment issues
        log += ["The following images were not aligned:"]

        for bad_file in bad_aligned_images:
            log += [f"\n\t{bad_file[:-5]}_reduced.fits"]

        bad_aligned_images = []

        master_aligned_images[object] = aligned_images

    return master_aligned_images

def create_fits(frame_info_df, master_aligned_images, output_dir, log):
    '''
    Creates and navigates to the new folder, then iterates through the dictionary of aligned images to get the headers
    associated with the original raw image. The data for each image  and headers are combined into a new fits file
    that is saved to the new directory. It then returns to the original directory.

    Args:
        data_dir: args.data, the directory with the data in it
        aligned_images: the list of aligned images
        log: List of strings containing logging to be added to the log txt file

    Returns:
        True (it completed)
    '''

    # Get the object names from the aligned data dictionary
    objects = list(master_aligned_images.keys())

    # Interpolate over the objects to save the aligned frames to the respective directories
    for object in objects:
        # Create sub directory to save the data
        log += ["\nCreating new directory...\n"]
        final_dir = os.path.join(output_dir, object)

        try:
            os.mkdir(final_dir)
        except OSError as error:
            print(error)

        # Define the aligned images for the respective object and get the file names
        aligned_images = master_aligned_images[object]
        file_names = list(aligned_images.keys())

        for file in tqdm(file_names, desc=f"Saving final images for {object}", unit=' frames',
                        dynamic_ncols=True):
            # Define the full path to where the respective aligned file is stored
            file_dir = frame_info_df[frame_info_df['Files'].isin([file])]['Directory'].iloc[0]
            file_path = os.path.join(file_dir, file)
            
            # For each frame in the raw, get the header
            raw_header = fits.getheader(file_path)

            # Create the file capsule
            hdu = fits.PrimaryHDU(data=aligned_images[file], header=raw_header)

            # Modify or remove the FWHM card from the header
            if 'FWHM' in hdu.header:
                del hdu.header['FWHM']
            
            # Create new file name "object.time.reduced.fits"
            file_name = file[:-5] + "_reduced.fits"

            # Write to the dir
            hdu.writeto(os.path.join(final_dir, file_name))

        log += ["Finished creating files for object " + object + "\n"]
        log += [f"Saving reduced frames to {final_dir}"]

    return True