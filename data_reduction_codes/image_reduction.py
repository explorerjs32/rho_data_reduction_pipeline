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

frame_info_df, observing_log_df = get_frame_info(args.light, args.dark, args.flat, args.bias)

light_objects = frame_info_df.loc[frame_info_df["Frame"] == "Light", "Object"].unique()

# Create the directory to save the images
output_dir = os.path.join(args.output, 'Reduced')
os.makedirs(output_dir, exist_ok=True)

for obj in light_objects:
    # Start of logging
    log = []

    exp_times = frame_info_df.loc[frame_info_df["Object"] == obj, "Exptime"].unique()
    filters =  frame_info_df.loc[frame_info_df["Object"] == obj, "Filter"].unique()
    
    log += ["RETRHO Data Reduction Pipeline Initiated...\n"
            "Reducing Data from " + "; ".join(args.light) + "\n\n"]

    log += ["Object observed: " + obj + "\n"
            "Light Frames: " + str(len(frame_info_df[(frame_info_df["Frame"] == "Light") & (frame_info_df["Object"] == obj)])) + "\n"]
    
    #Printing the amount of frames for an object at each exposure time
    for filter in frame_info_df.loc[frame_info_df["Object"] == obj, "Filter"].unique():
        #print(filter)
        integration_time = 0
        for time in exp_times:
            if len(frame_info_df[(frame_info_df['Object'] == obj) & (frame_info_df['Exptime'] == time) & (frame_info_df['Filter'] == filter)]) > 0:
                #print(f"    Exposed {str(len(frame_info_df[(frame_info_df['Object'] == obj) & (frame_info_df['Exptime'] == time) & (frame_info_df['Filter'] == filter)]))} frames at {time} seconds \n")
                log += ["   Filter " + filter + ": \n"
                    f"      Exposed {str(len(frame_info_df[(frame_info_df['Object'] == obj) & (frame_info_df['Exptime'] == time) & (frame_info_df['Filter'] == filter)]))} frames at {time} seconds \n"]
                int_time = len(frame_info_df[(frame_info_df['Object'] == obj) & (frame_info_df['Exptime'] == time) & (frame_info_df['Filter'] == filter)]) * time / 60
                integration_time += int_time
        log += [f"      Total Integration for {filter}: {integration_time} minutes \n"]

    #Need to only list calibration frames used for the object
    obj_darks = 0
    for time in exp_times:
        count = (len(frame_info_df[(frame_info_df['Exptime'] == time) & (frame_info_df['Frame'] == "Dark")]))
        obj_darks += count

    obj_flats = 0
    for filter in filters:
        count = (len(frame_info_df[(frame_info_df['Filter'] == filter) & (frame_info_df['Frame'] == "Flat")]))
        obj_flats += count

    log += [
        "Dark Frames: " + str(obj_darks) + "\n"
        "Flat Frames: " + str(obj_flats) + "\n"
        "Bias Frames: " + str(frame_info_df["Frame"].str.count("Bias").sum()) + "\n\n"]

    if bool(args.background_subtract):
        log += ["Background Subtraction is ON\n\n"]

    elif not bool(args.background_subtract):
        log += ["Background Subtraction is OFF\n\n"]

    # Still need to make it so the log only writes the creation of master cailbration frames used for this objects exposure times and filters
    # Identify master bias frames and combine them
    log += ["Creating Master Bias...\n"]
    master_bias, master_bias_noise = create_master_bias(frame_info_df, log)
    log += ["Done!\n\n"]

    # Create the master darks
    log += ["Creating Master Darks...\n"]
    dark_times, master_darks, uncertainties_dark_current = create_master_darks(frame_info_df, master_bias, log, exp_times)
    log += ["Done!\n\n"]

    # Create master flats
    log += ["Creating Master Flats...\n"]
    flat_filters, master_flats, flats_uncertainty_dict = create_master_flats(frame_info_df, dark_times, master_darks, master_bias, master_bias_noise, uncertainties_dark_current, log, filters)
    log += ["Done!\n\n"]

    print("Done creating Calibration frames\n")

    # Conduct image reduction process with new dataframe only including one object
    obj_frame_info_df = frame_info_df[frame_info_df["Object"] == obj]
    reduced_images = image_reduction(obj_frame_info_df, dark_times, master_darks, flat_filters, master_flats, master_bias, log, bool(args.background_subtract))
    print("Done reducing the raw data\n")

    # Aligned the reduced images
    aligned_images = align_images(reduced_images, log)
    print("Done aligning images\n")

    # Create fits images and extract the information on reduced frames
    reduced_frames_df = create_fits(frame_info_df, aligned_images, output_dir, log, master_bias_noise, uncertainties_dark_current, flats_uncertainty_dict)
    print("Done with the data reduction. See final report on " + output_dir)

    # Transferring log list of strings to the txt file
    logfile = open(output_dir + f"/{obj}/{obj}_data_reduction_report.txt", "a")
    for line in log:
        logfile.write(line)
    logfile.close()