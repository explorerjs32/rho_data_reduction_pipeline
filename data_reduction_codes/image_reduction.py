import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_threshold, detect_sources
from photutils.utils import circular_footprint
from scipy.ndimage import shift
from reduction_functions import * # type: ignore
import os
import argparse
from tqdm.auto import tqdm


# Define the arguments to parse into the script
parser = argparse.ArgumentParser(
    description="Arguments to parse for the data reduction pipeline. Primarily foccusing on the directories where the data is stored.")

parser.add_argument('-l', '--light', type=str, nargs='+', required=True, help="Single or multiple directories containing light frames only")
parser.add_argument('-d', '--dark', type=str, nargs='+', required=True, help="Single or multiple directories containing dark frames only")
parser.add_argument('-f', '--flat', type=str, nargs='+', required=True, help="Single or multiple directories containing flat frames only")
parser.add_argument('-b', '--bias', type=str, nargs='+', required=True, help="Single or multiple directories containing bias frames only")
parser.add_argument('-B', '--background_subtract', type=str2bool, default=True, help="Perform background subtraction (True/False). Default is True")
parser.add_argument('-o', '--output', type=str, default='', required=True, help='Output directory to store the reduced frames. This directory MUST BE pre-created.')

args = parser.parse_args()

print(f"RETRHO Data Reduction Pipeline Initiated...\nReducing Data from " + "; ".join(args.light) + "\n")

# Start of logging
log = []
log += ["RETRHO Data Reduction Pipeline Initiated...\n"
        "Reducing Data from " + "; ".join(args.light) + "\n\n"]

frame_info_df, observing_log_df = get_frame_info(args.light, args.dark, args.flat, args.bias)
log += ["Objects observed: " + str(len(frame_info_df["Object"].unique()) - 1) + "\n"
        "Light Frames: " + str(frame_info_df["Frame"].str.count("Light").sum()) + "\n"
        "Dark Frames: " + str(frame_info_df["Frame"].str.count("Dark").sum()) + "\n"
        "Flat Frames: " + str(frame_info_df["Frame"].str.count("Flat").sum()) + "\n"
        "Bias Frames: " + str(frame_info_df["Frame"].str.count("Bias").sum()) + "\n\n"]

if bool(args.background_subtract):
    log += ["Background Subtraction is ON\n\n"]

elif not bool(args.background_subtract):
    log += ["Background Subtraction is OFF\n\n"]


# Identify master bias frames and combine them
log += ["Creating Master Bias...\n"]
master_bias, master_bias_noise = create_master_bias(frame_info_df, log)
log += ["Done!\n\n"]

# Create the master darks
log += ["Creating Master Darks...\n"]
dark_times, master_darks, uncertainties_dark_current = create_master_darks(frame_info_df, master_bias_noise, log)
log += ["Done!\n\n"]

# Create master flats
log += ["Creating Master Flats...\n"]
flat_filters, master_flats, flats_uncertainty_dict = create_master_flats(frame_info_df, dark_times, master_darks, master_bias, log)
log += ["Done!\n\n"]

print("Done creating Calibration frames\n")

# Conduct image reduction process
reduced_images = image_reduction(frame_info_df, dark_times, master_darks, flat_filters, master_flats, master_bias, log, bool(args.background_subtract))
print("Done reducing the raw data\n")

# Aligned the reduced images
aligned_images = align_images(reduced_images, log)
print("Done aligning images\n")

# Create the directory to save the images
output_dir = os.path.join(args.output, 'Reduced')
os.makedirs(output_dir, exist_ok=True)

# Create fits images and extract the information on reduced frames
reduced_frames_df = create_fits(frame_info_df, aligned_images, output_dir, log)
print("Done with the data reduction. See final report on " + output_dir)

# Transferring log list of strings to the txt file
logfile = open(output_dir + "/data_reduction_report.txt", "a")
for line in log:
    logfile.write(line)
logfile.close()

# Saving data of normalized flat, bias noise, and dark current
uncertainties = [("Read_Noise", master_bias_noise)]  # Read noise row
uncertainties.extend(uncertainties_dark_current.items())  # Dark current rows
uncertainties.extend(flats_uncertainty_dict.items())  # Flat noise rows

df = pd.DataFrame(uncertainties)
csv_path = os.path.join(output_dir, 'Uncertainties.csv')
df.to_csv(csv_path, index=False, header=False, sep=" ", quoting=3)  # 'sep=" "' ensures space separation isntead of comma, quoting=3 ensures no quotes around the strings

# Saving the information on reduced frames
reduced_frames_df.to_csv(os.path.join(output_dir, 'Reduced_Frames_info.csv'), index=False, sep=" ")  # 'sep=" "' ensures space separation isntead of comma, quoting=3 ensures no quotes around the strings