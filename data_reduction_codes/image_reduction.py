import numpy as np
import pandas as pd
from astropy.io import fits
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
        obj_name = fits.getheader(f"{data_dir}{file}")['OBJECT']
        objects.append(obj_name)

        # Get the frame type
        frame = fits.getheader(f"{data_dir}{file}")['FRAME']
        frames.append(frame)

        # Get the exposure time 
        exp_time = fits.getheader(f"{data_dir}{file}")['EXPTIME']
        exposure_times.append(exp_time)

        # Get the filter used for the exposure
        filter = fits.getheader(f"{data_dir}{file}")['FILTER']
        filters.append(filter)

    # Generate a dataframe containing the frame information
    frame_info_df = pd.DataFrame({'Files':file_list,
                                'Object':objects,
                                'Frame':frames, 
                                'Filter':filters, 
                                'Exptime':exposure_times})
    
    # Generate the observing log based on the frame information
    observing_log_df = frame_info_df.groupby(by=['Object','Frame','Filter','Exptime']).size().to_frame(name='Exposures').reset_index()

    return frame_info_df, observing_log_df


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
    biases_data = np.array([fits.getdata(data_dir + file).astype(float) for file in biases_df["Files"].values])

    # Using median combine to form a final master bias frame and then return it
    master_bias = np.median(biases_data, axis=0)
    return master_bias


# Define the arguments to parse into the script
parser = argparse.ArgumentParser(description="Arguments to parse for the data reduction pipeline. Primarily foccusing on the directories where the data is stored.")

parser.add_argument('-D', '--data', type=str, required=True, help="Directory where the collected data is stored.")
parser.add_argument('-b', '-bias_frames', type=str, default='', help="Directory where the bias frames are stored.")
parser.add_argument('-d','--dark_frames', type=str, default='', help="Directory where the dark frames are stored.")
parser.add_argument('-f','--flat_frames', type=str, default='', help="Directory where the flat frames are stored.")
parser.add_argument('-l', '--light_frames', type=str, default='', help="Directory where the light (science) frames are stored.")

args = parser.parse_args()


# Extract the frame information from the collected data and the observing log
frame_info_df, observing_log_df = get_frame_info(args.data, os.listdir(args.data))

# Identify master bias frames and combine them
master_bias = create_master_bias(frame_info_df, args.data)
