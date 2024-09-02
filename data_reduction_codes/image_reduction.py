import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_threshold, detect_sources
from photutils.utils import circular_footprint
from astropy.visualization import ZScaleInterval, ImageNormalize
from skimage.registration import *
from scipy.ndimage import interpolation as interp
import os
import argparse


# Define functions here
def get_frame_info(data_dir):
    """
    Extracts information from FITS file headers and returns it as pandas DataFrames.


    This function reads the FITS files located in the specified directory, extracts relevant
    information from the headers (object name, frame type, exposure time, and filter), and
    compiles this information into a pandas DataFrame. Additionally, it generates an observing
    log DataFrame based on the extracted frame information.


    Parameters:
    data_dir (str): The directory where the FITS files are stored.
    file_list (list of str): A list of FITS file names to be processed.


    Returns:
    tuple:
        pandas.DataFrame: A DataFrame containing the extracted information with the following columns:
                          'Files' - the FITS file names,
                          'Object' - the object names from the FITS headers,
                          'Frame' - the frame types from the FITS headers,
                          'Filter' - the filters used for the exposures from the FITS headers,
                          'Exptime' - the exposure times from the FITS headers.
        pandas.DataFrame: An observing log DataFrame grouped by 'Object', 'Frame', 'Filter', and 'Exptime'
                          with a column 'Exposures' indicating the number of exposures for each group.
    """

    # Get the list of fits files in the directory
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.fits') and os.path.isfile(os.path.join(data_dir, f))]

    # Define the lists to store the data
    exposure_times = []
    filters = []
    frames = []
    objects = []

    # Loop through the light frames to get the information out of the fits file header
    for file in file_list:
        # Get the object name
        obj_name = fits.getheader(os.path.join(data_dir, file))['OBJECT']
        objects.append(obj_name)

        # Get the frame type
        frame = fits.getheader(os.path.join(data_dir, file))['FRAME']
        frames.append(frame)

        # Get the exposure time
        exp_time = fits.getheader(os.path.join(data_dir, file))['EXPTIME']
        exposure_times.append(exp_time)

        # Get the filter used for the exposure
        filter = fits.getheader(os.path.join(data_dir, file))['FILTER']
        filters.append(filter)

    # Generate a dataframe containing the frame information
    frame_info_df = pd.DataFrame({'Files': file_list,
                                  'Object': objects,
                                  'Frame': frames,
                                  'Filter': filters,
                                  'Exptime': exposure_times})

    # Generate the observing log based on the frame information
    observing_log_df = frame_info_df.groupby(by=['Object', 'Frame', 'Filter', 'Exptime']).size().to_frame(
        name='Exposures').reset_index()

    return frame_info_df, observing_log_df


def create_master_darks(frame_info_df):
    """
    Creates a list of master darks from the information in the two dataframes.

    First, isolates the dark frames and how many unique exposures there are, then iterates through
    the list of darks for each exposure time to gather the dark frames for a specific exposure time, which
    it will median combine into a master dark.

    Args:
        frame_info_df: the frame information dataframe
        observing_log_df: the dataframe list of unique frame types


    Returns:
        dark_exposure_times: a list of master dark exposure times (float) that correlate to the master darks
        master_darks: a dictionary of master darks. Each object in the list is fits data.
            Key: "master_darks_[0.0]s" where [0.0] is replaced with the exposure time
            Value: fits data (2D array of pixel counts)
    """
    # creating the master darks- one for each exposure time.]
    # for each unique exposure (entry in observing log that is a dark frame), get that exposure time
    darks_df = frame_info_df[frame_info_df['Frame'] == 'Dark'].reset_index(drop=True)
    dark_exposure_times = darks_df['Exptime'].unique()

    # go through the darks of that exposure length to create the master-
    master_darks = {}
    for exp in dark_exposure_times:
        darks_exp = []
        for index, row in darks_df.iterrows():
            if (row["Exptime"] == exp):
                darks_exp.append(fits.getdata(os.path.join(args.data, row['Files'])))
        master_darks["master_dark_" + str(exp) + "s"] = np.median(np.array(darks_exp), axis=0)
        # Logging master darks created
        print("Master_dark_" + str(int(exp)) + "s created")

    # return the darks and the times they correlate to.
    return dark_exposure_times, master_darks


def create_master_bias(frame_info_df, data_dir):
    '''

    Identifies bias frames, compiles and returns them using numpy median method

    Using the data extracted from fits by get_frame_info, this function compiles the various
    frames that have the bias identification into biases_files. It then proceeds to collect
    the data of each of the frames into biases_data. Finally, the collection of data is then
    combined using the numpy median method into the master_bias variable. That is then
    returned.

    Args:
        frame_info_df: (Pandas DF list) Collection of frame data at given directory
        data_dir: (Str) Path leading to the directory desired for analysis

    Returns:
        master_bias: (2D array of integers) Median of master bias data used for subsequent
        calculations

    '''

    # Filtering dataframes that are labeled as bias using df indexing
    biases_df = frame_info_df[frame_info_df["Frame"] == "Bias"].reset_index(drop=True)

    # Expanding data within dataframes of label bias into an array
    biases_data = np.array([fits.getdata(os.path.join(data_dir, file)).astype(float) for file in biases_df["Files"].values])

    # Using median combine to form a final master bias frame and then return it
    master_bias = np.median(biases_data, axis=0)
    return master_bias


def create_master_flats(frame_info_df, data_dir, darks_exptimes, master_darks, master_bias):
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
            flat_exptime = fits.getheader(os.path.join(data_dir, row['Files']))['EXPTIME']

            if flat_exptime in darks_exptimes and fits.getheader(os.path.join(data_dir, row['Files']))[
                'FILTER'] == filter_name:
                # print(f"subtracting {flat_exptime}s master dark from {row['Files']}")
                flats_filter.append(
                    fits.getdata(os.path.join(data_dir, row['Files'])) - master_darks[f"master_dark_{flat_exptime}s"])

            elif flat_exptime not in darks_exptimes and fits.getheader(os.path.join(data_dir, row['Files']))[
                'FILTER'] == filter_name:
                # print(f"subtracting master bias from {row['Files']}")
                flats_filter.append(fits.getdata(os.path.join(data_dir, row['Files'])) - master_bias)

        # Combine the flats and normalize the master flat
        master_flat = np.median(np.array(flats_filter), axis=0)
        normalized_master_flat = master_flat / np.median(master_flat)
        master_flats["master_flat_" + filter_name] = normalized_master_flat
        # Logging Master flat creation
        print("Master_flat_" + filter_name + " created")

    return flat_filters, master_flats

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


def image_reduction(frame_info_df, dark_times, master_darks, flat_filters, master_flats, master_bias, data_dir):
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

    Returns:
        dict: Dictionary of reduced images, keyed by the original file names. Each value is a 2D numpy array representing 
              the reduced image, with the background subtracted and bad pixels masked.
    """

    # Initializes a dataframe containing all the information on raw images
    raw_image_df = frame_info_df[frame_info_df["Frame"] == "Light"].reset_index(drop=True)

    # Initialize the image reduced final product
    reduced_images = {}

    # create the flat masks from reduced flats
    def bad_pixel(pixel_data):
        return True if (pixel_data > 2 or pixel_data < 0.5) else False

    bad_pixel_v = np.vectorize(bad_pixel)
    masks = {}
    for i in range(len(flat_filters)):
        masks["mask_" + flat_filters[i]] = bad_pixel_v(master_flats["master_flat_" + flat_filters[i]])

    # Iterate through files and create individual reduced data images
    for index in range(len(raw_image_df)):

        # Identify current file, file data, and the current exposure time
        file = raw_image_df["Files"][index]
        raw_image_data = fits.getdata(os.path.join(data_dir, file))
        raw_image_exp_time = raw_image_df["Exptime"][index]
        obj_name = fits.getheader(os.path.join(data_dir, file))['OBJECT']

        # Logging image reduction object name, file, exposure time
        print("Reducing raw light frames for " + obj_name)
        print("Raw file: " + file)
        print("Exposure time: " + str(int(raw_image_exp_time)) + " sec")

        # Identify dark_frame match OR closest match
        dark_frame = []
        # If identical match found use that given current dark
        if "master_dark_" + str(raw_image_exp_time) + "s" in master_darks:
            dark_frame = master_darks["master_dark_" + str(raw_image_exp_time) + "s"]
            # Logging master dark subtraction
            print("Subtracted Master_dark_" + str(raw_image_exp_time) + "s")
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
            print("Subtracted Master_dark_" + str(dark_times[min(range(len(dark_times)), key=lambda i: abs(dark_times[i]-temp_times[0]))]) + "s")

        # Identify flat_frame match OR provide feedback on missing filters for master_flats
        flat_frame = []
        flat_frame_found = False
        # FIXME - Fix filter issue
        if "master_flat_" + raw_image_df["Filter"][index] in master_flats:
            # If filter key found, use the given flat_frame
            flat_frame = master_flats["master_flat_" + raw_image_df["Filter"][index]]
            flat_frame_found = True
            # Logging master flat division
            print("Divided normalized_master_flat_" + raw_image_df["Filter"][index])
        else:
            # Otherwise, identify the missing filters
            print("Filter Error: missing filter " + raw_image_df["Filter"][index] + " for file " + raw_image_df["Files"][index] +
                  ". Frame type = " + raw_image_df["Frame"][index])

        # Perform image reduction now that everything is in place (if statement required for missing filter errors)
        if flat_frame_found and obj_name != 'Unknown':
            # Reduce the light frames
            reduced_image = (raw_image_data - dark_frame) / flat_frame

            # Subtract the background of the reduced image
            bkg_subtracted_reduced_image = background_subtraction(reduced_image)

            # mask the bad pixels (MBP) in the background subtracted reduced science images
            np.putmask(bkg_subtracted_reduced_image, masks["mask_" + raw_image_df["Filter"][index]], -999)
            final_reduced_image = bkg_subtracted_reduced_image
            
            # Store the final reduced image into the list
            reduced_images[file] = final_reduced_image

        # Logging FIXME - Needs to be updated with new functions
        print("Removed bad pixels")
        print("Subtracted background")
        print(file + " reduced!\n")

    # Finally, return the image reduced product
    return reduced_images


def align_images(reduced_data):
    """
    Aligns a series of reduced astronomical images to a template image using phase cross-correlation.

    This function takes a dictionary of reduced images, identifies the brightest pixel in the first image (used as a template),
    and then aligns the remaining images to this template. Alignment is performed by determining the shift needed to match
    the template image, using phase cross-correlation, and then applying this shift to each image.

    Args:
        reduced_data (dict): Dictionary containing image data, where the keys are file names and the values are 2D numpy arrays 
                             representing the reduced images.

    Returns:
        dict: A dictionary of aligned images, where the keys are the original file names and the values are the aligned 2D numpy 
              arrays. The first image in the list is returned unaltered, as it serves as the template for alignment.
    """
    # Split the image data and the file names into lists
    images = list(reduced_data.values())
    files = list(reduced_data.keys())

    # Define the template image to allign the rest to
    # This template image is always the first image of the input list
    template_image = images[0]

    # Find the (y,x) brightest pixel coordinate
    ypix, xpix = np.unravel_index(template_image.argmax(), template_image.shape)

    # Based on the (y,x) pixel coordinates further clip the template image using a 100 pixel window
    template_image_xy_clip = template_image[ypix - 50:ypix + 50, xpix - 50:xpix + 50]

    # Define a list to store the aligned images
    aligned_images = {}

    # Interpolate over the list of images to align them
    for i, image in enumerate(images):

        # If the image is the first one and the template image, add it to the list
        if i == 0:
            aligned_images[files[i]] = image

        # Else, align the rest of the images to the template
        elif i > 0:
            # Clip the rest of the images using the 100 pixel window around the template's (y,x) brightest pixel coordinate
            target_image_xy_clip = image[ypix - 50:ypix + 50, xpix - 50:xpix + 50]

            # Calculate the target image shift with respect to the template image
            shift_vals, error, diffphase = phase_cross_correlation(template_image_xy_clip, target_image_xy_clip)

            # print(f"Image {i} shift values {shift_vals}")

            # Align the images and add them to the list
            aligned_image = interp.shift(image, shift_vals)
            aligned_images[files[i]] = aligned_image

    return aligned_images


def create_fits(data_dir, aligned_images):
    '''
    Creates and navigates to the new folder, then iterates through the dictionary of aligned images to get the headers
    associated with the original raw image. The data for each image  and headers are combined into a new fits file
    that is saved to the new directory. It then returns to the original directory.

    Args:
        data_dir: args.data, the directory with the data in it
        aligned_images: the list of aligned images

    Returns:
        True (it completed)
    '''

    # create new directory
    print("Creating new directory...")
    new_dir_path = os.path.join(data_dir, "reduced_images")
    try:
        os.mkdir(new_dir_path)
    except OSError as error:
        print(error)

    #navigate into the new directory
    print("Changing directories...")
    orig_dir = os.getcwd()
    os.chdir(new_dir_path)

    for file in aligned_images:
        print(file)
        # for each frame in the raw, get the header
        raw_header = fits.getheader(os.path.join(orig_dir, data_dir, file))
        # create the file capsule
        hdu = fits.PrimaryHDU(data=aligned_images[file], header=raw_header)
        # create new file name "object.time.reduced.fits"
        file_name = file[:-4] + "reduced.fits"
        # write to the dir
        hdu.writeto(file_name)
    print("Finished creating files")
    os.chdir(orig_dir)
    return True


# Define the arguments to parse into the script
parser = argparse.ArgumentParser(
    description="Arguments to parse for the data reduction pipeline. Primarily foccusing on the directories where the data is stored.")

parser.add_argument('-D', '--data', type=str, required=True, help="Directory where the collected data is stored.")
parser.add_argument('-b', '-bias_frames', type=str, default='', help="Directory where the bias frames are stored.")
parser.add_argument('-d', '--dark_frames', type=str, default='', help="Directory where the dark frames are stored.")
parser.add_argument('-f', '--flat_frames', type=str, default='', help="Directory where the flat frames are stored.")
parser.add_argument('-l', '--light_frames', type=str, default='', help="Directory where the light (science) frames are stored.")

args = parser.parse_args()

# Start of logging
print()
print("RETRHO Data Reduction Pipeline Initiated...\n"
      "Reducing Data from " + args.data + "\n")

# Extract the frame information from the collected data and the observing log
frame_info_df, observing_log_df = get_frame_info(args.data)

print("Objects observed: " + str(len(frame_info_df["Object"].unique())) + "\n"
      "Light Frames: " + str(frame_info_df["Frame"].str.count("Light").sum()) + "\n"
      "Dark Frames: " + str(frame_info_df["Frame"].str.count("Dark").sum()) + "\n"
      "Flat Frames: " + str(frame_info_df["Frame"].str.count("Flat").sum()) + "\n"
      "Bias Frames: " + str(frame_info_df["Frame"].str.count("Bias").sum()) + "\n")

# Identify master bias frames and combine them
print("Creating Master Bias...")
master_bias = create_master_bias(frame_info_df, args.data)
print("Done!\n")

# create the master darks
print("Creating Master Darks...")
dark_times, master_darks = create_master_darks(frame_info_df)
print("Done!\n")

# Create master flats
print("Creating Master Flats...")
flat_filters, master_flats = create_master_flats(frame_info_df, args.data, dark_times, master_darks, master_bias)
print("Done!\n")

# Conduct image reduction process
reduced_images = image_reduction(frame_info_df, dark_times, master_darks, flat_filters, master_flats, master_bias, args.data)

# Aligned the reduced images
print("Saving reduced frames to ") #FIXME - Directory for frames saving
print("Done reducing image data. See final report on ") #FIXME - Directory for report saving
aligned_images = align_images(reduced_images)


#create fits images
create_fits(args.data, aligned_images)

