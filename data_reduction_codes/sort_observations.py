import numpy as np
import pandas as pd
from astropy.io import fits
import os
import shutil
import argparse
import warnings


def get_frame_info(directories):
    """
    Extracts information from FITS file headers across multiple directories and returns it as pandas DataFrames.

    This function reads the FITS files located in the specified directories, extracts relevant
    information from the headers (object name, frame type, exposure time, and filter), and
    compiles this information into a pandas DataFrame. Additionally, it generates an observing
    log DataFrame based on the extracted frame information.

    Parameters:
    directories (list of str): List of directories where the FITS files are stored.

    Returns:
    tuple:
        pandas.DataFrame: A DataFrame containing the extracted information with the following columns:
                          'Files' - the FITS file names,
                          'Directory' - the directory where the file is stored,
                          'Object' - the object names from the FITS headers,
                          'Frame' - the frame types from the FITS headers,
                          'Filter' - the filters used for the exposures from the FITS headers,
                          'Exptime' - the exposure times from the FITS headers.
        pandas.DataFrame: An observing log DataFrame grouped by 'Object', 'Frame', 'Filter', 'Exptime'
                          with a column 'Exposures' indicating the number of exposures for each group.
    """

    # Define the lists to store the data
    exposure_times = []
    filters = []
    frames = []
    objects = []
    file_list = []
    directories_list = []

    # Loop through all provided directories
    for data_dir in directories:
        # Get the list of fits files in the current directory
        current_file_list = [f for f in os.listdir(data_dir) if f.endswith('.fits') and os.path.isfile(os.path.join(data_dir, f))]

        # Loop through each file in the directory and extract header info
        for file in current_file_list:
            # Get the object name from the FITS header
            header = fits.getheader(os.path.join(data_dir, file))
            obj_name = header.get('OBJECT', 'Unknown')

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

def organize_frames(data_dir, frame_info_df):
    """
    Organizes the FITS files into sub-directories based on their frame type and object name.

    This function creates sub-directories for each frame type (Light, Dark, Bias, Flat) and
    stores the FITS files accordingly. For light frames, it creates additional sub-directories
    for each object.

    Parameters:
    frame_info_df (pandas.DataFrame): A DataFrame containing the frame information.

    Returns:
    None
    """

    # Create sub-directories for each frame type
    for frame_type in frame_info_df['Frame'].unique().astype(str):
        frame_type_dir = os.path.join(data_dir, frame_type)
        os.makedirs(frame_type_dir, exist_ok=True)

    # Process light frames
    light_frames = frame_info_df[frame_info_df['Frame'] == 'Light']

    # Store the files of each respective object into a dictionary
    light_frames_per_object = {}

    for object_name in light_frames['Object'].unique():
        if object_name == 'Unknown':
            pass

        elif light_frames['Object'].unique().size == 1 and object_name == 'Unknown':
            warnings.warn(f"No Light frames were collected for any specific object")


        else:
            object_frames = light_frames[light_frames['Object'] == object_name]

            # Create a sub-directory for the object inside the 'Light' directory
            object_dir = os.path.join(os.path.join(data_dir, 'Light'), object_name)
            os.makedirs(object_dir, exist_ok=True)

            # Store the light frames per object on a list
            files_out = []
            
            for file in object_frames['Files']:
                shutil.copy2(os.path.join(data_dir, file), os.path.join(object_dir, file))
                # print(f"{file} is a light frame for object {object_name}")
                files_out.append(file)

            light_frames_per_object[object_name] = files_out
            files_out = []

    # Look for the calibration frames of each object's light frame
    for object_name in light_frames_per_object.keys():

        # Clasify the bias frames
        bias_dir = os.path.join(data_dir, 'Bias')
        bias_frames_files = frame_info_df[frame_info_df['Frame'] == 'Bias']['Files']

        if bias_frames_files.size == 0:
            warnings.warn(f"Missing Bias frames")

        else:
            for file in bias_frames_files:
                shutil.copy2(os.path.join(data_dir, file), os.path.join(bias_dir, file))
                # print(f"{file} is a flat frame for object {object_name}")   

        # Get the filters of each object
        # filters = frame_info_df[frame_info_df['Files'].isin(light_frames_per_object[object_name])]['Filter'].unique()
        filters = frame_info_df[frame_info_df['Frame'] == 'Flat']['Filter'].unique()

        # Find the corresponding flat frames
        flats_dir = os.path.join(data_dir, 'Flat')
        flat_frames_out = []

        for filt in filters:
            flat_frames_files = frame_info_df[((frame_info_df['Frame'] == 'Flat') & (frame_info_df['Filter'] == filt))]['Files']

            if flat_frames_files.size == 0:
                warnings.warn(f"Missing Flat frames for object {object_name} with filter {filt}")

            else:
                for file in flat_frames_files:
                    shutil.copy2(os.path.join(data_dir, file), os.path.join(flats_dir, file))
                    flat_frames_out.append(file)
                    # print(f"{file} is a flat frame for object {object_name}")             

        # Find the corresponding light and flat frames
        # exp_times = frame_info_df[(frame_info_df['Files'].isin(light_frames_per_object[object_name])) & \
        #                           (frame_info_df['Files'].isin(flat_frames_out))]['Exptime'].unique()
        exp_times = frame_info_df[(frame_info_df['Files'].isin(flat_frames_out))]['Exptime'].unique().tolist()
        exp_times += frame_info_df[(frame_info_df['Files'].isin(light_frames_per_object[object_name]))]['Exptime'].unique().tolist()

        # Find the corresponding dark frames
        darks_dir = os.path.join(data_dir, 'Dark')

        for exptime in exp_times:
            dark_frames_files = frame_info_df[((frame_info_df['Frame'] == 'Dark') & (frame_info_df['Exptime'] == exptime))]['Files']
            
            if dark_frames_files.size == 0:
                warnings.warn(f"Missing Dark frames for object {object_name} with exposure time of {exptime} seconds.")

            else:
                for file in dark_frames_files:
                    shutil.copy2(os.path.join(data_dir, file), os.path.join(darks_dir, file))
                    # print(f"{file} is a dark frame for object {object_name}")
                     

    return

# Define the arguments to be parsed into the script
parser = argparse.ArgumentParser(
    description="Argumets to parse into the observations organizer script. It focuses on organizing the fits files from an input directory containing observations from the Rosemary Hill Observatory.")

parser.add_argument('-dir', '--dir', type=str, nargs='+', required=True, help="Single or multiple directories where observations (fits files) are stored.")

args = parser.parse_args()


# Extract the observing log and classified information from the fits files
frame_info_df, observing_log_df = get_frame_info(args.dir)

print(frame_info_df)
print(observing_log_df)

organize_frames(args.dir[0], frame_info_df)