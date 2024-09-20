import numpy as np
import pandas as pd
import os
from astropy.io import fits
import argparse


# Define the functions

# Add get_frame_info() function

# Add source_detection() function

# Add aperture_photometry() function

# Add psf_photometry() function


# Define the arguments to parse into the code
parser = argparse.ArgumentParser(
    description="Arguments to parse for the Photometric Analysis. Parse in the data directory where the fully reduced data is stored.")

parser.add_argument('-data', '--data', type=str, required=True, help="Directory where the reduced data is stored.")
parser.add_argument('-output', '--output', type=str, default='', help='Output directory to store the photometric calculations.')

args = parser.parse_args()
