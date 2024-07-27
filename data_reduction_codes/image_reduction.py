import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize
import os
import argparse


# Define functions here
def get_frame_info(data_dir, file_list):
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


def create_master_flats(frame_info_df, data_dir):
    """
    Creates a list of normalized master flats for each filter from the frame information dataframe.

    First, isolates the flat frames and iterates through the list of flats for each filter to gather the flat frames.
    It then median-combines these frames to create a master flat for each filter, normalizing each master flat.

    Args:
        frame_info_df: DataFrame containing information about the frames, including 'Frame' type and 'Filter'.
        data_dir: String representing the directory where the flat frames are stored.

    Returns:
        flat_filters: A list of unique filter names present in the frame information dataframe.
        master_flats: A dictionary of master flats. Each key is a string "master_flat_[FilterName]" and each value
            is a 2D numpy array representing the normalized master flat for that filter.
    """

    # Isolate the flat frames from the dataframe
    flats_df = frame_info_df[frame_info_df['Frame'] == 'Flat'].reset_index(drop=True)
    flat_filters = flats_df['Filter'].unique()

    # Create the master flats
    master_flats = {}
    for filter_name in flat_filters:
        flats_filter = []
        for index, row in flats_df.iterrows():
            if row["Filter"] == filter_name:
                file_path = os.path.join(data_dir, row['Files'])
                flats_filter.append(fits.getdata(file_path))

        # Combine the flats and normalize the master flat
        master_flat = np.median(np.array(flats_filter), axis=0)
        normalized_master_flat = master_flat / np.median(master_flat)
        master_flats["master_flat_" + filter_name] = normalized_master_flat

    return flat_filters, master_flats

def image_reduction(frame_info_df, master_darks, master_bias, data_dir):
    """

    Isolates the raw images and subtract the master_dark for image reduction

    The dataframes containing the light images are isolated into

    Args:
        frame_info_df
        master_darks
        master_bias
        data_dir

    Returns:
         bias_removed_dark_subtracted_light_frames - A collection of images which remove the master_darks and bias
            affecting the image and skewing the data.

    """

    # Collect all raw images
    raw_image_df = frame_info_df[frame_info_df["Frame"] == "Light"].reset_index(drop=True)

    # Extract data using np.array method
    raw_image_data = np.array([fits.getdata(data_dir + file).astype(float) for file in raw_image_df["Files"].values])

    # Build the dark subtracted data array
    # Key error, don't know exactly how to access proper key in master_darks (data from raw image df?)
    dark_subtracted_light_frames = np.array((light_image / master_darks["Master_Darks_" + "s"]) for light_image in raw_image_data)

    # Conduct removal of bias from images
    bias_removed_dark_subtracted_light_frames = np.array((dark_subtracted_light_image - master_bias) for dark_subtracted_light_image in dark_subtracted_light_frames)

    return bias_removed_dark_subtracted_light_frames

def image_reduction(frame_info_df, master_darks, master_bias, data_dir):
    """

    Isolates the raw images and subtract the master_dark for image reduction

    The dataframes containing the light images are isolated into

    Args:
        frame_info_df
        master_darks
        master_bias
        data_dir

    Returns:
         bias_removed_dark_subtracted_light_frames - A collection of images which remove the master_darks and bias
            affecting the image and skewing the data.

    """

    # Collect all raw images
    raw_image_df = frame_info_df[frame_info_df["Frame"] == "Light"].reset_index(drop=True)

    # Extract data using np.array method
    raw_image_data = np.array([fits.getdata(data_dir + file).astype(float) for file in raw_image_df["Files"].values])

    # Build the dark subtracted data array
    # Key error, don't know exactly how to access proper key in master_darks (data from raw image df?)
    dark_subtracted_light_frames = np.array((light_image / master_darks["Master_Darks_" + "s"]) for light_image in raw_image_data)

    # Conduct removal of bias from images
    bias_removed_dark_subtracted_light_frames = np.array((dark_subtracted_light_image - master_bias) for dark_subtracted_light_image in dark_subtracted_light_frames)

    return bias_removed_dark_subtracted_light_frames


# Define the arguments to parse into the script
parser = argparse.ArgumentParser(
    description="Arguments to parse for the data reduction pipeline. Primarily foccusing on the directories where the data is stored.")

parser.add_argument('-D', '--data', type=str, required=True, help="Directory where the collected data is stored.")
parser.add_argument('-b', '-bias_frames', type=str, default='', help="Directory where the bias frames are stored.")
parser.add_argument('-d', '--dark_frames', type=str, default='', help="Directory where the dark frames are stored.")
parser.add_argument('-f', '--flat_frames', type=str, default='', help="Directory where the flat frames are stored.")
parser.add_argument('-l', '--light_frames', type=str, default='',
                    help="Directory where the light (science) frames are stored.")

args = parser.parse_args()


# Extract the frame information from the collected data and the observing log
frame_info_df, observing_log_df = get_frame_info(args.data, os.listdir(args.data))

# Identify master bias frames and combine them
master_bias = create_master_bias(frame_info_df, args.data)

# create the master darks
dark_times, master_darks = create_master_darks(frame_info_df)

# Identify master bias frames and combine them
master_bias = create_master_bias(frame_info_df, args.data)

# Conduct image reduction process
reduced_image = image_reduction(frame_info_df, master_darks, master_bias, args.data)

# Create master flats
flat_filters, master_flats = create_master_flats(frame_info_df, args.data)
