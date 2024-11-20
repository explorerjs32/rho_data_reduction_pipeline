import numpy as np
import pandas as pd
import os
from astropy.io import fits
import argparse
from photutils.detection import DAOStarFinder
from photutils import aperture_photometry, CircularAperture, CircularAnnulus, datasets, find_peaks


# Define the functions

# Add get_frame_info() function
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

# Add source_detection() function

# Add aperture_photometry() function

# Add psf_photometry() function


# Define the arguments to parse into the code
parser = argparse.ArgumentParser(
    description="Arguments to parse for the Photometric Analysis. Parse in the data directory where the fully reduced data is stored.")

parser.add_argument('-data', '--data', type=str, required=True, help="Directory where the reduced data is stored.")
parser.add_argument('-output', '--output', type=str, default='', help='Output directory to store the photometric calculations.')
parser.add_argument('-reduced', '--reduced_dirs', type=str, nargs='+', help="Single or multiple directories containing reduced frames.")

args = parser.parse_args()

if args.reduced_dirs:
    print(f"Parsing reduced frames from directories: {', '.join(args.reduced_dirs)}")
    reduced_frame_info = get_frame_info(args.reduced_dirs)

    # Save the results to a CSV file
    reduced_frame_info.to_csv('reduced_frame_info.csv', index=False)
    print("Reduced frame information saved to 'reduced_frame_info.csv'")
    print(reduced_frame_info)