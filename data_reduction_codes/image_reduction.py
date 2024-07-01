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

    ''' NOTE: inside this function, getheader uses [data_dir][file], which assumes that data_dir (args.data) has
        "/" at the end of it, like "data/2024-04-15/" rather than "data/2024-04-15". If the latter is entered
        as a command line argument, the program will search for files in the data folder like "data/2024-04-15rho..."
        instead of "data/2024-04-15/rho..." which obviously does not exist and results in an error.
 
        This is just something to keep in mind when running it and later on if we work on a way to automate running
        this program. Optionally, we could have a function that will add the "/" at the end of the string if it is
        missing, but for now it's not a big deal.
    '''

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
    frame_info_df = pd.DataFrame({'Files': file_list,
                                  'Object': objects,
                                  'Frame': frames,
                                  'Filter': filters,
                                  'Exptime': exposure_times})

    # Generate the observing log based on the frame information
    observing_log_df = frame_info_df.groupby(by=['Object', 'Frame', 'Filter', 'Exptime']).size().to_frame(
        name='Exposures').reset_index()

    return frame_info_df, observing_log_df


def create_master_darks(frame_info_df, observing_log_df):
    """
    Creates a list of master darks from the information in the two dataframes.


    First, makes of list of how many different dark exposure times there are, then iterates through
    the list of files for each exposure time to gather the dark frames for a specific exposure time, which
    it will median combine into a master dark.


    NOTE: this could probably be done with dictionaries. This does work, but for compatibility reasons should we
    choose to prioritize dictionaries, I will rewrite this to use dictionaries (or create another function version
    that uses dictionaries)


    Args:
        frame_info_df: the frame information dataframe
        observing_log_df: the dataframe list of unique frame types


    Returns:
        dark_exposure_times: a list of master dark exposure times (float) that correlate to the master darks
        master_darks: a list of master darks. Each object in the list is fits data.


    """
    # creating the master darks- one for each exposure time.
    # practice dataset has 20 15sec darks and 5 100sec darks - access info from observing_log

    # for each unique exposure (entry in observing log that is a dark frame), get that exposure time
    dark_exposure_times = []
    for index, row in observing_log_df.iterrows():
        # create list of unique dark exposure times
        if (row['Frame'] == 'Dark'):
            dark_exposure_times.append(row['Exptime'])

    '''go through the other dataframes and grab the ones of that exposure length to create the master-
           this could probably be optimized, it currently iterates through frame_info_df len(dark_exposure_times) times,
           the other option is iterating through dark_exposure_times len(frame_info_df) times,
           I wonder if there's a way to only have to iterate through both 1 time, I just don't know currently.
    '''
    master_darks = []
    for exp in dark_exposure_times:
        darks_exp = []
        # put the data from each of these dark frames into the list
        for index, row in frame_info_df.iterrows():
            if (row["Frame"] == 'Dark' and row["Exptime"] == exp):
                darks_exp.append(fits.getdata(f"{args.data}{row['Files']}"))
        # have all the dark frame data for exp time, combine into a master dark:
        master_darks.append(np.median(np.array(darks_exp), axis=0))

    # return the darks and the times they correlate to.
    return dark_exposure_times, master_darks


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

# create the master darks
dark_times, master_darks = create_master_darks(frame_info_df, observing_log_df)
