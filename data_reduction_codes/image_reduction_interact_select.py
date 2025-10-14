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

import tkinter as tk
from tkinter import filedialog, messagebox, BooleanVar

"""
An interactive selection tool for the image reduction step of the pipeline. 
Functionally the exact same as image_reduction.py, but rather than manually inputting directories in the command line, 
a UI appears to select the path of each frame type and specify the desired output directory for your reduced frames. 
A check box is included to toggle the optional background subtraction reduction step. 
Requires tkinter package. 

- Note: For usage after having run sort_observations.py"""
class DirectorySelectorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Data Reduction Setup")

        self.light_dir = tk.StringVar()
        self.dark_dir = tk.StringVar()
        self.flat_dir = tk.StringVar()
        self.bias_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.background_subtract = BooleanVar(value=True)

        # Add buttons and labels
        self.create_directory_selector("Light Frames", self.light_dir, 0)
        self.create_directory_selector("Dark Frames", self.dark_dir, 1)
        self.create_directory_selector("Flat Frames", self.flat_dir, 2)
        self.create_directory_selector("Bias Frames", self.bias_dir, 3)
        self.create_directory_selector("Output Directory", self.output_dir, 4)

        # Background subtract checkbox
        tk.Checkbutton(master, text="Background Subtraction", variable=self.background_subtract).grid(row=5, column=0, sticky='w', padx=10, pady=10)

        # Submit button
        tk.Button(master, text="Submit", command=self.submit).grid(row=6, column=0, columnspan=2, pady=10)

    # def create_directory_selector(self, label, var, row):
    #     tk.Label(self.master, text=label).grid(row=row, column=0, sticky='w', padx=10, pady=5)
    #     tk.Button(self.master, text="Select", command=lambda: self.select_directory(var, label)).grid(row=row, column=1, padx=10, pady=5)

    def create_directory_selector(self, label, var, row):
        tk.Label(self.master, text=label).grid(row=row, column=0, sticky='w', padx=10, pady=5)

        # Status label to confirm selection
        status_label = tk.Label(self.master, text="Not selected", fg="red")
        status_label.grid(row=row, column=2, sticky='w', padx=10)

        # Pass label to update after directory selection
        tk.Button(self.master, text="Select", command=lambda: self.select_directory(var, label, status_label)).grid(row=row, column=1, padx=10, pady=5)

    def select_directory(self, var, label, status_label):
    
        start_dir = var.get() if var.get() else os.getcwd()
        directory = filedialog.askdirectory(title=f"Select {label}", initialdir=start_dir)
        if directory:
            var.set(directory)
            # Show path relative to current working directory
            rel_path = os.path.relpath(directory, os.getcwd())
            display_path = f"{rel_path}"
            status_label.config(text=display_path, fg="green")
            print(f"{label} selected: {directory}")
        else:
            status_label.config(text="Not selected", fg="red")

    # def select_directory(self, var, label):
    #     # directory = filedialog.askdirectory(title=f"Select {label}")
    #     # directory = filedialog.askdirectory(title=f"Select {label}", initialdir=os.path.abspath(os.sep))
    #     start_dir = var.get() if var.get() else os.path.abspath(os.getcwd())
    #     directory = filedialog.askdirectory(title=f"Select {label}", initialdir=start_dir)
    #     if directory:
    #         var.set(directory)
    #         print(f"{label} selected: {directory}")

    def submit(self):
        # Validate that all directories are selected
        missing = []
        if not self.light_dir.get(): missing.append("Light")
        if not self.dark_dir.get(): missing.append("Dark")
        if not self.flat_dir.get(): missing.append("Flat")
        if not self.bias_dir.get(): missing.append("Bias")
        if not self.output_dir.get(): missing.append("Output")

        if missing:
            messagebox.showerror("Missing Directories", f"Please select: {', '.join(missing)}")
            return

        # Close the window
        self.master.destroy()

    def get_args(self):
        return {
            "light": [self.light_dir.get()],
            "dark": [self.dark_dir.get()],
            "flat": [self.flat_dir.get()],
            "bias": [self.bias_dir.get()],
            "output": self.output_dir.get(),
            "background_subtract": self.background_subtract.get()
        }


def main():
    root = tk.Tk()
    app = DirectorySelectorApp(root)
    root.mainloop()

    # Get the selected arguments
    args = app.get_args()
    print("\nSelected Arguments:")
    for k, v in args.items():
        print(f"{k}: {v}")

    # If you want an object-style interface:
    class Args: pass
    args_obj = Args()
    for key, val in args.items():
        setattr(args_obj, key, val)

    return args_obj


if __name__ == "__main__":
    args = main()




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
flat_filters, master_flats, flats_uncertainty_dict = create_master_flats(frame_info_df, dark_times, master_darks, master_bias, master_bias_noise, uncertainties_dark_current, log)
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
reduced_frames_df = create_fits(frame_info_df, aligned_images, output_dir, log, master_bias_noise, uncertainties_dark_current, flats_uncertainty_dict)
print("Done with the data reduction. See final report on " + output_dir)

# Transferring log list of strings to the txt file
logfile = open(output_dir + "/data_reduction_report.txt", "a")
for line in log:
    logfile.write(line)
logfile.close()

# Code to save uncertainties to a separate csv file 
# Saving data of normalized flat, bias noise, and dark current
# uncertainties = [("Read_Noise", master_bias_noise)]  # Read noise row
# uncertainties.extend(uncertainties_dark_current.items())  # Dark current rows
# uncertainties.extend(flats_uncertainty_dict.items())  # Flat noise rows

# df = pd.DataFrame(uncertainties)
# csv_path = os.path.join(output_dir, 'Uncertainties.csv')
# df.to_csv(csv_path, index=False, header=False, sep=" ")  # 'sep=" "' ensures space separation isntead of comma, quoting=3 ensures no quotes around the strings

# Saving the information on reduced frames
reduced_frames_df.to_csv(os.path.join(output_dir, 'Reduced_Frames_info.csv'), index=False, sep=" ")  # 'sep=" "' ensures space separation isntead of comma, quoting=3 ensures no quotes around the strings