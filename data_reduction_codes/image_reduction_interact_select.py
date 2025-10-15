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

Does require running on individual objects, as of now.

- Note: For usage after having run sort_observations.py"""
class DirectorySelectorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Data Reduction Setup")

        self.light_dir = tk.StringVar()
        self.additional_light_dirs = []  # List to store additional light frame directories
        self.dark_dir = tk.StringVar()
        self.flat_dir = tk.StringVar()
        self.bias_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.background_subtract = BooleanVar(value=True)
        
        # Store widget references for repositioning
        self.dark_widgets = []
        self.flat_widgets = []
        self.bias_widgets = []
        self.output_widgets = []
        self.add_button = None
        self.bg_checkbox = None
        self.submit_button = None
        
        self.current_row = 0

        # Add buttons and labels
        self.create_directory_selector("Light Frames", self.light_dir, self.current_row)
        self.current_row += 1
        
        # Add button for additional light frames
        self.add_button = tk.Button(master, text="+ Add Another Light Directory", command=self.add_light_directory)
        self.add_button.grid(row=self.current_row, column=0, columnspan=2, pady=5)
        self.additional_light_row = self.current_row
        self.current_row += 1
        
        self.dark_widgets = self.create_directory_selector("Dark Frames", self.dark_dir, self.current_row)
        self.current_row += 1
        self.flat_widgets = self.create_directory_selector("Flat Frames", self.flat_dir, self.current_row)
        self.current_row += 1
        self.bias_widgets = self.create_directory_selector("Bias Frames", self.bias_dir, self.current_row)
        self.current_row += 1
        self.output_widgets = self.create_directory_selector("Output Directory", self.output_dir, self.current_row)
        self.current_row += 1

        # Background subtract checkbox
        self.bg_checkbox = tk.Checkbutton(master, text="Background Subtraction", variable=self.background_subtract)
        self.bg_checkbox.grid(row=self.current_row, column=0, sticky='w', padx=10, pady=10)
        self.current_row += 1

        # Submit button
        self.submit_button = tk.Button(master, text="Submit", command=self.submit)
        self.submit_button.grid(row=self.current_row, column=0, columnspan=2, pady=10)

    # def create_directory_selector(self, label, var, row):
    #     tk.Label(self.master, text=label).grid(row=row, column=0, sticky='w', padx=10, pady=5)
    #     tk.Button(self.master, text="Select", command=lambda: self.select_directory(var, label)).grid(row=row, column=1, padx=10, pady=5)

    def create_directory_selector(self, label, var, row):
        label_widget = tk.Label(self.master, text=label)
        label_widget.grid(row=row, column=0, sticky='w', padx=10, pady=5)

        # Status label to confirm selection
        status_label = tk.Label(self.master, text="Not selected", fg="red")
        status_label.grid(row=row, column=2, sticky='w', padx=10)

        # Pass label to update after directory selection
        button_widget = tk.Button(self.master, text="Select", command=lambda: self.select_directory(var, label, status_label))
        button_widget.grid(row=row, column=1, padx=10, pady=5)
        
        # Return widget references for later repositioning
        return [label_widget, button_widget, status_label]

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

    def add_light_directory(self):
        """Add an additional light frame directory"""
        # Create a new StringVar for the additional directory
        new_light_var = tk.StringVar()
        
        # Insert the new row after the add button
        insert_row = self.additional_light_row + len(self.additional_light_dirs) + 1
        
        # Create label and button for the additional light directory
        label_widget = tk.Label(self.master, text=f"Light Frames {len(self.additional_light_dirs) + 2}")
        label_widget.grid(row=insert_row, column=0, sticky='w', padx=10, pady=5)
        
        # Status label
        status_label = tk.Label(self.master, text="Not selected", fg="red")
        status_label.grid(row=insert_row, column=2, sticky='w', padx=10)
        
        # Select button
        button_widget = tk.Button(self.master, text="Select", command=lambda: self.select_directory(new_light_var, f"Additional Light Frames {len(self.additional_light_dirs) + 2}", status_label))
        button_widget.grid(row=insert_row, column=1, padx=10, pady=5)
        
        # Store the StringVar, status label, and widgets
        self.additional_light_dirs.append((new_light_var, status_label, [label_widget, button_widget, status_label]))
        
        # Reposition all widgets below the inserted row
        self.reposition_widgets()
        
    def reposition_widgets(self):
        """Reposition all widgets after adding a new light directory"""
        # Calculate the new row for the add button
        add_button_row = self.additional_light_row + len(self.additional_light_dirs) + 1
        self.add_button.grid(row=add_button_row, column=0, columnspan=2, pady=5)
        
        # Reposition dark, flat, bias, output widgets
        dark_row = add_button_row + 1
        for widget in self.dark_widgets:
            widget.grid(row=dark_row)
            
        flat_row = dark_row + 1
        for widget in self.flat_widgets:
            widget.grid(row=flat_row)
            
        bias_row = flat_row + 1
        for widget in self.bias_widgets:
            widget.grid(row=bias_row)
            
        output_row = bias_row + 1
        for widget in self.output_widgets:
            widget.grid(row=output_row)
            
        # Reposition background checkbox
        bg_row = output_row + 1
        self.bg_checkbox.grid(row=bg_row)
        
        # Reposition submit button
        submit_row = bg_row + 1
        self.submit_button.grid(row=submit_row)
        
        # Update current_row
        self.current_row = submit_row

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
        
        # If output directory is not selected, use parent directory of light directory
        if not self.output_dir.get():
            parent_dir = os.path.dirname(self.light_dir.get())
            parenter_dir = os.path.dirname(parent_dir)
            self.output_dir.set(parenter_dir)
            print(f"Output directory not selected. Using parent directory: {parent_dir}")

        if missing:
            messagebox.showerror("Missing Directories", f"Please select: {', '.join(missing)}")
            return

        # Close the window
        self.master.destroy()

    def get_args(self):
        # Collect all light directories (main + additional)
        all_light_dirs = [self.light_dir.get()]
        for light_var, status_label, widgets in self.additional_light_dirs:
            if light_var.get():  # Only add if a directory was actually selected
                all_light_dirs.append(light_var.get())
        
        return {
            "light": all_light_dirs,
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