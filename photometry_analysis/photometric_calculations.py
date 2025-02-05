import numpy as np
import pandas as pd
import os
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import argparse
from photutils.detection import DAOStarFinder
from photutils import aperture_photometry, CircularAperture, CircularAnnulus, datasets, find_peaks

def get_frame_info(directories):
    """
    Extracts information from FITS file headers for reduced frames across multiple directories.

    Parameters:
    directories (list of str): List of directories containing reduced FITS frames.

    Returns:
    pandas.DataFrame: A DataFrame with the following columns:
                      - 'Directory': Directory name.
                      - 'File': File name.
                      - 'Object': Object name (header keyword 'OBJECT').
                      - 'Date-Obs': Observation date and time (header keyword 'DATE-OBS').
                      - 'Filter': Filter used (header keyword 'FILTER').
                      - 'Exptime': Exposure time in seconds (header keyword 'EXPTIME').
    """
    # Define lists to store extracted data
    directories_list = []
    file_list = []
    objects = []
    dates = []
    filters = []
    exposure_times = []

    # Iterate through all directories
    for directory in directories:
        # Get all FITS files in the directory
        fits_files = [f for f in os.listdir(directory) if f.endswith('.fits')]

        for file in fits_files:
            try:
                # Read the FITS header
                header = fits.getheader(os.path.join(directory, file))

                # Extract relevant header information
                directories_list.append(directory)
                file_list.append(file)
                objects.append(header.get('OBJECT', 'Unknown'))
                dates.append(header.get('DATE-OBS', 'Unknown'))
                filters.append(header.get('FILTER', 'Unknown'))
                exposure_times.append(header.get('EXPTIME', 'Unknown'))
            except Exception as e:
                print(f"Error processing file {file} in {directory}: {e}")
                continue

    # Create a DataFrame with the extracted information
    reduced_frame_info = pd.DataFrame({
        'Directory': directories_list,
        'File': file_list,
        'Object': objects,
        'Date-Obs': dates,
        'Filter': filters,
        'Exptime': exposure_times
    })

    return reduced_frame_info

def image_stats_out(image_data_path):
    """Calculates and returns statistics of an astronomical image.

    Loads an astronomical image from a FITS file, computes basic statistics 
    (mean, median, and standard deviation) using sigma-clipping, and returns 
    the image data and calculated statistics.

    Args:
        image_data_path (str): The path to the FITS file containing the image data.

    Returns:
        tuple: A tuple containing the following:
        - image_data (numpy.ndarray): The image data as a NumPy array.
        - mean (float): The sigma-clipped mean of the image data.
        - median (float): The sigma-clipped median of the image data.
        - stddev (float): The sigma-clipped standard deviation of the image data.
    """
    # Load up the image data
    image_data = fits.getdata(image_data_path)

    # Compute the initial statistics of the image
    mean, median, stddev = sigma_clipped_stats(image_data, sigma=3.0)

    return image_data, mean, median, stddev

def find_star_peaks(image_data, threshold, median, std):
    """
    Finds peaks in the image data and returns a DataFrame with peak information.

    Args:
    image_data: The 2D NumPy array representing the image.
    threshold: The threshold value for peak finding (in standard deviations).
    median: The median value of the image data.
    std: The standard deviation of the image data.

    Returns:
    A pandas DataFrame containing peak positions and IDs.
    """
    # Define the threshold to find the star peak positions
    threshold_abs = median + (threshold * std)
 
    # Find the peak of the stars
    tbl = find_peaks(image_data, threshold=threshold_abs, box_size=11)
 
    if tbl is not None:
        tbl = tbl.to_pandas().sort_values(by='peak_value', ascending=False).iloc[:].reset_index(drop=True)
        tbl['id'] = tbl.index + 1
        return tbl
    else:
        return pd.DataFrame(columns=['x_peak', 'y_peak', 'id']) 
