import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.widgets import Button, RectangleSelector, RadioButtons, CheckButtons, Slider, AxesWidget
import matplotlib.gridspec as gridspec
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ImageNormalize, ZScaleInterval
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.modeling import models, fitting
import warnings
from astropy.utils.exceptions import AstropyUserWarning
import astropy.units as u
import argparse
import os
from tqdm import tqdm
from scipy import ndimage
from PyQt5.QtWidgets import QApplication
import tkinter as tk
from tkinter import filedialog, messagebox, BooleanVar
import sys


class Compute_PSF_Photometry:
    def __init__(self, frame_info_df, uncertainties_df, num_bins=None, reference_image=None, reference_stars=None):
        """
        Initializes the photometry computation class, incorporating optional image binning 
        and multi-filter reference plotting.
        """
        self.frame_info = frame_info_df
        self.uncertainties_df = uncertainties_df
        self.gain = 0.37
        self.num_bins = num_bins
        self.reference_image = reference_image
        self.reference_stars = reference_stars
        self.images = {}
        self.load_all_images()
        self.star_positions = {}  # {star_number: (x, y)}
        self.photometry = {}      # {filename: {star_number: flux}
        self.results_df = None
        
        self.current_star = 1
        
        # Get screen dimensions using Qt
        app = QApplication.instance() or QApplication(sys.argv)
        screen = app.primaryScreen()
        screen_geometry = screen.availableGeometry()
        screen_height = screen_geometry.height()
        screen_width = screen_geometry.width()

        # Calculate figure size to match screen height (with some padding)
        dpi = 100.0  # matplotlib default DPI
        height_inches = (screen_height * 0.85) / dpi  # 85% of screen height
        width_inches = height_inches * 1.0  # 1:1 aspect ratio for photometry tool
        
        # Close any existing figures
        plt.close('all')
        
        # Create the figure with calculated size
        self.fig = plt.figure(num='PSF Photometry Tool', figsize=(width_inches, height_inches))
        
        # Note: self.ax creation is moved to setup_plot() to handle dynamic subplots
        
        # Position window at the center of the screen
        manager = plt.get_current_fig_manager()
        if hasattr(manager, 'window'):
            window = manager.window
            x = (screen_width - window.width()) // 2
            y = (screen_height - window.height()) // 2
            window.move(x, y)
        
        self.setup_plot()
        self.create_widgets()
        
        # Connect zoom event
        self.scroll_cid = self.fig.canvas.mpl_connect('scroll_event', self.zoom_image)

    def zoom_image(self, event):
        """
        Zoom in and out of the image using the scroll wheel on whichever axis is hovered.
        """
        is_main_ax = event.inaxes == self.ax
        is_ref_ax = hasattr(self, 'ax_ref') and event.inaxes == self.ax_ref
        
        if is_main_ax or is_ref_ax:
            ax = event.inaxes
            scale_factor = 1.25 if event.button == 'down' else 0.75
            
            # Get current x and y limits
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            
            # Calculate the new limits
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            
            xcenter = (cur_xlim[0] + cur_xlim[1]) / 2
            ycenter = (cur_ylim[0] + cur_ylim[1]) / 2
            
            ax.set_xlim([xcenter - new_width / 2, xcenter + new_width / 2])
            ax.set_ylim([ycenter - new_height / 2, ycenter + new_height / 2])
            
            # Get the original dimensions of the specific image hovered
            img_shape = self.image_data.shape if is_main_ax else self.reference_image.shape
            image_width = img_shape[1]
            image_height = img_shape[0]
            
            # Set the limits to the original image dimensions if zoomed out too much
            if new_width > image_width:
                ax.set_xlim(0, image_width)
                
            if new_height > image_height:
                ax.set_ylim(0, image_height)
                
            self.fig.canvas.draw_idle()

    def load_all_images(self):
        """Load all images into memory."""
        num_images = len(self.frame_info)
        
        # Scenario A: 1 bin per image (No binning)
        if self.num_bins is None or self.num_bins == num_images:
            for idx, row in self.frame_info.iterrows():
                filepath = os.path.join(row['Directory'], row['File'])
                with fits.open(filepath) as hdul:
                    self.images[row['File']] = hdul[0].data.astype(float)
                    
        # Scenario B: Image binning requested (num_bins < num_images)
        else:
            images_per_bin = num_images // self.num_bins
            remaining_images = num_images % self.num_bins
            
            start_idx = 0
            binned_records = []
            
            for i in range(self.num_bins):
                # Calculate end index: distribute remainder across the first few bins
                end_idx = start_idx + images_per_bin
                if i < remaining_images:
                    end_idx += 1
                    
                group_df = self.frame_info.iloc[start_idx:end_idx]
                coadded_data = None
                
                # Loop through images in the current bin and sum their data arrays
                for idx, row in group_df.iterrows():
                    filepath = os.path.join(row['Directory'], row['File'])
                    with fits.open(filepath) as hdul:
                        data = hdul[0].data.astype(float)
                        if coadded_data is None:
                            coadded_data = data
                        else:
                            coadded_data += data  # Co-adding by summing the arrays
                
                # Use the first file's name in the bin as the representative key
                representative_filename = group_df.iloc[0]['File']
                self.images[representative_filename] = coadded_data
                
                # Keep the representative record for the binned image and log bin stats
                record = group_df.iloc[0].copy()
                record['File'] = representative_filename
                record['Bin_Number'] = i + 1
                record['Images_In_Bin'] = len(group_df)
                binned_records.append(record)
                
                # Advance the start index for the next bin
                start_idx = end_idx
                
            # Replace the original frame_info with the binned summary.
            self.frame_info = pd.DataFrame(binned_records)

    def setup_plot(self):
        """Initialize the plot with the first image, and optionally a reference image."""
        first_file = self.frame_info['File'].iloc[0]
        self.image_data = self.images[first_file]
        
        # If we have a reference image from a previous filter, create two stacked subplots
        if self.reference_image is not None and self.reference_stars is not None:
            self.ax_ref = self.fig.add_subplot(211)  # 2 rows, 1 col, top plot
            self.ax = self.fig.add_subplot(212)      # 2 rows, 1 col, bottom plot
            
            # Plot reference image on the top
            ref_norm = ImageNormalize(self.reference_image, interval=ZScaleInterval())
            self.ax_ref.imshow(self.reference_image, origin='lower', cmap='gray', norm=ref_norm)
            self.ax_ref.set_title("Reference Filter Stars")
            
            # Draw the previously selected stars on the reference image
            for star_num, (x, y) in self.reference_stars.items():
                self.ax_ref.plot(x, y, 'rx', markersize=8)
                self.ax_ref.text(x + 15, y + 15, f'Star {star_num}', color='red', fontsize=10)
            self.ax_ref.axis('off')
            
            # Plot current image on the bottom
            image_norm = ImageNormalize(self.image_data, interval=ZScaleInterval())
            self.ax.imshow(self.image_data, origin='lower', cmap='gray', norm=image_norm)
            self.ax.set_title(f"Current Filter: {self.frame_info['Filter'].iloc[0]}")
            
        else:
            # First filter scenario: single plot
            self.ax = self.fig.add_subplot(111)
            image_norm = ImageNormalize(self.image_data, interval=ZScaleInterval())
            self.ax.imshow(self.image_data, origin='lower', cmap='gray', norm=image_norm)
        
        # Add image info text box anchored to the bottom of the active/current axis
        textstr = (f"File: {first_file}\n"
                  f"Object: {self.frame_info['Object'].iloc[0]}\n"
                  f"Exposure Time: {self.frame_info['Exptime'].iloc[0]} secs\n"
                  f"Filter: {self.frame_info['Filter'].iloc[0]}")
                  
        # Move the text box below the bottom left of the image (y = -0.10 relative to the axis)
        self.ax.text(0.02, -0.10, textstr, transform=self.ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='top')

        self.ax.axis('off')
        
        # Adjust figure margins to ensure the text box and 'Done' button don't overlap
        plt.subplots_adjust(bottom=0.25)

    def create_widgets(self):
        """Create the interface widgets."""
        # Rectangle selector for star selection
        self.rect_selector = RectangleSelector(
            self.ax, self.on_select,
            useblit=True,
            props=dict(facecolor='red', edgecolor='red', alpha=0.2),
            interactive=True
        )
        
        # Done button
        self.done_button_ax = plt.axes([0.13, 0.05, 0.25, 0.04])
        self.done_button = Button(self.done_button_ax, 'Done with Star Selection')
        self.done_button.on_clicked(self.finish_selection)
        
        # Initially disable button
        self.done_button.set_active(False)

    def on_select(self, eclick, erelease):
        """Handle rectangle selection."""
        # Check if both click and release events are within the axes
        if eclick is None or erelease is None:
            return
        if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
            return
            
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        # Check if the selection has a valid size
        if abs(x2-x1) < 1 or abs(y2-y1) < 1:
            return
        
        # Extract region and find peak
        region = self.image_data[
            min(y1,y2):max(y1,y2),
            min(x1,x2):max(x1,x2)
        ]
        
        # Check if region is not empty
        if region.size == 0:
            return
            
        local_y, local_x = np.unravel_index(np.argmax(region), region.shape)
        
        # Calculate absolute peak position
        peak_x = min(x1,x2) + local_x
        peak_y = min(y1,y2) + local_y
        
        # Store star position
        self.star_positions[self.current_star] = (peak_x, peak_y)
        
        # Plot marker and label
        self.ax.plot(peak_x, peak_y, 'rx', markersize=10)
        self.ax.text(peak_x + 15, peak_y + 15, f'Star {self.current_star}',
                    color='red', fontsize=12)
        
        self.current_star += 1
        self.fig.canvas.draw_idle()
        
        # Enable done button when at least one star is selected
        self.done_button.set_active(True)

    def finish_selection(self, event):
        """Complete star selection and close the figure."""
        plt.close(self.fig)
        self.results_df = self.compute_all_photometry() 

    def compute_all_photometry(self):
        """Compute photometry for all images and display results."""
        print("\nComputing photometry for all images...")
        
        # Compute photometry for all images
        for filename in tqdm(self.images.keys(), desc="Processing images"):
            image = self.images[filename]
            self.photometry[filename] = {}
            
            # Get image metadata from frame_info
            frame_row = self.frame_info[self.frame_info['File'] == filename].iloc[0]
            exptime = frame_row['Exptime']
            filter_ = frame_row['Filter']
            
            # Get timestamp from frame info - assuming there's a DATE-OBS or similar field
            # If it's not in frame_info, you may need to extract it from FITS headers
            if 'DATE-OBS' in frame_row:
                date_obs = frame_row['DATE-OBS']
            else:
                # Try to get from FITS header
                filepath = os.path.join(frame_row['Directory'], frame_row['File'])
                date_obs = fits.getheader(filepath).get('DATE-OBS', '')
            
            # Calculate BJD if we have a timestamp
            if date_obs:
                try:
                    # Convert to Time object
                    t = Time(date_obs, format='isot', scale='utc')
                    
                    # Get target coordinates - assuming we have RA and DEC in frame_info
                    # If not in frame_info, extract from FITS header
                    if 'RA' in frame_row and 'DEC' in frame_row:
                        ra = frame_row['RA']
                        dec = frame_row['DEC']
                    else:
                        header = fits.getheader(filepath)
                        ra = header.get('RA', 0.0)
                        dec = header.get('DEC', 0.0)
                    
                    # Create SkyCoord object for the target
                    target_coord = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))
                    
                    # Get observer location (you may need to set this for your observatory)
                    observatory = EarthLocation(lat=29.4001*u.deg, lon=-82.5862*u.deg, height=23*u.m)
                    
                    # Calculate BJD
                    bjd = t.tdb + t.light_travel_time(target_coord, location=observatory)
                    
                    # Store BJD in photometry dictionary
                    self.photometry[filename]['BJD'] = bjd.jd
                except Exception as e:
                    print(f"Warning: Could not calculate BJD for {filename}: {e}")
                    self.photometry[filename]['BJD'] = None
            else:
                self.photometry[filename]['BJD'] = None
            
            # Compute photometry for each star
            for star_num, (x, y) in self.star_positions.items():
                # Extract region around star
                size = 40
                half_size = size // 2
                star_cutout = image[
                    max(0, y-half_size):min(image.shape[0], y+half_size),
                    max(0, x-half_size):min(image.shape[1], x+half_size)
                ]
                
                # Calculate statistics and threshold
                mean, median, std = sigma_clipped_stats(star_cutout, sigma=3.0)
                threshold = median + 3.0 * std
                
                # Create initial mask of pixels above threshold
                bright_pixels = star_cutout > threshold
                
                # Label connected regions
                labeled_regions, num_regions = ndimage.label(bright_pixels)
                
                # Define center coordinates in cutout
                center_y, center_x = half_size, half_size
                
                # Find region containing center
                center_region = labeled_regions[center_y, center_x]
                
                # Create X, Y coordinate grids for the 40x40 cutout
                y_grid, x_grid = np.mgrid[:star_cutout.shape[0], :star_cutout.shape[1]]

                # Estimate initial guesses for the fitter to converge faster
                sky_guess = np.median(star_cutout)
                amp_guess = np.max(star_cutout) - sky_guess
                
                # Define the model: A 2D Constant (Sky) + A 2D Gaussian (Star)
                # Note: Even if you ignore sky noise, you MUST fit the sky pedestal, 
                # otherwise the Gaussian amplitude will be artificially inflated.
                g_init = (models.Const2D(amplitude=sky_guess) + 
                          models.Gaussian2D(amplitude=amp_guess, 
                                            x_mean=half_size, y_mean=half_size, 
                                            x_stddev=2.0, y_stddev=2.5,
                                            theta=0.1))

                # Initialize the Levenberg-Marquardt least-squares fitter
                fitter = fitting.LevMarLSQFitter()

                # Fit the model to the data 
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', AstropyUserWarning)
                    fit_model = fitter(g_init, x_grid, y_grid, star_cutout)

                # Extract the Gaussian star component (index 1, since Const2D is index 0)
                psf_model = fit_model[1]
                
                # Calculate Total Flux (The volume under the 2D Gaussian)
                # Formula: Volume = 2 * pi * Amplitude * sigma_x * sigma_y
                flux_adu = 2 * np.pi * psf_model.amplitude.value * psf_model.x_stddev.value * psf_model.y_stddev.value
                
                # Calculate the "Effective Area" of the PSF for the noise equation
                # Formula: N_eff = 4 * pi * sigma_x * sigma_y
                npix_eff = 4 * np.pi * psf_model.x_stddev.value * psf_model.y_stddev.value

                # Calculate uncertainties
                read_noise = self.uncertainties_df.loc['Read_Noise', 'Value']
                dark_current = self.uncertainties_df.loc[f'Dark_Current_{exptime}s', 'Value']
                flat_noise = self.uncertainties_df.loc[f'Flat_{filter_}_Noise', 'Value']
                gain = self.gain

                # Calculate flux and store results
                flux_out = flux_adu* gain
                self.photometry[filename][f"Flux_Star_{star_num}"] = flux_out
                
                # Calculate flux uncertainty
                flux_noise = np.sqrt(
                    (flux_out) + # Shot noise
                    (npix_eff * dark_current * gain) +  # Dark current noise
                    (npix_eff * (flat_noise * gain)**2.) +  # Flat field noise
                    (npix_eff * (read_noise * gain)**2)  # Read noise
                )
                self.photometry[filename][f"Flux_err_Star_{star_num}"] = flux_noise
                
                # Calculate instrumental magnitude and error
                minst = -2.5 * np.log10(flux_out/exptime)
                minst_err = (2.5 / np.log(10)) * (flux_noise / flux_out)
                
                self.photometry[filename][f"Minst_Star_{star_num}"] = minst
                self.photometry[filename][f"Minst_err_Star_{star_num}"] = minst_err
                
                # Store coordinates
                self.photometry[filename][f"Star_{star_num}_x"] = x
                self.photometry[filename][f"Star_{star_num}_y"] = y

        # Create results DataFrame
        data = []
        for filename in self.images.keys():
            row = {'File': filename}
            # Add BJD as second column
            if 'BJD' in self.photometry[filename]:
                row['BJD'] = self.photometry[filename].pop('BJD')  # Remove from dict after getting value
            else:
                row['BJD'] = None
            # Add all other photometry data
            row.update(self.photometry[filename])
            data.append(row)
        
        # Create DataFrame with columns in desired order
        results_df = pd.DataFrame(data)
        
        # Ensure BJD is the second column after File
        if 'BJD' in results_df.columns:
            # Get all columns except File and BJD
            other_cols = [col for col in results_df.columns if col not in ['File', 'BJD']]
            # Reorder columns with File first, BJD second, then all others
            results_df = results_df[['File', 'BJD'] + other_cols]
        
        return results_df
    
class DirectorySelectorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Input directory for PSF Photometry")
        self.reduced_dir = tk.StringVar()
        self.output_dir = tk.StringVar()

        self.dp_widgets = []
        self.output_widgets = []
        
        self.current_row = 0

        # Add buttons and labels for data directory selection
        self.create_directory_selector("Reduced data directory", self.reduced_dir, self.current_row)
        self.current_row += 1
        self.create_directory_selector("Output directory", self.output_dir, self.current_row)
        self.current_row += 1
        # Submit button
        self.submit_button = tk.Button(master, text="Submit", command=self.submit)
        self.submit_button.grid(row=self.current_row, column=0, columnspan=2, pady=10)

    def create_directory_selector(self, label, var, row):
        label_widget = tk.Label(self.master, text=label)
        label_widget.grid(row=row, column=0, sticky='w', padx=10, pady=5)

        # Status label to confirm selection
        status_label = tk.Label(self.master, text="Not selected", fg="red")
        status_label.grid(row=row, column=2, sticky='w', padx=10)

        # Pass label to update after directory selection
        button_widget = tk.Button(self.master, text="Select", command=lambda: self.select_directory(var, label, status_label))
        button_widget.grid(row=row, column=1, padx=10, pady=5)

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

    def submit(self):
        #validate directories
        missing=[]
        if not self.reduced_dir.get():
            missing.append("Missing reduced data directory")
        
        # If output directory is not selected, default to current working directory
        if not self.output_dir.get():
            self.output_dir.set(os.getcwd())
            print("No output directory selected. Defaulting to current working directory.")

        # Check if required files exist in the reduced data directory
        reduced_dir = self.reduced_dir.get()
        frame_info_file = os.path.join(reduced_dir, 'frame_info.csv')
        uncertainties_file = os.path.join(reduced_dir, 'uncertainties.csv')
        
        if not os.path.exists(frame_info_file):
            missing.append("frame_info.csv not found in reduced data directory, run image_reduction scripts before performing photometry")
        if not os.path.exists(uncertainties_file):
            missing.append("uncertainties.csv not found in reduced data directory, run image_reduction scripts before performing photometry")

        if missing:
            messagebox.showerror("Error", f"Errors:\n- " + "\n- ".join(missing))
            return

        self.master.destroy()

    def get_args(self):
        return {"d":self.reduced_dir.get(), "o":self.output_dir.get()}

def main():
    root = tk.Tk()
    app = DirectorySelectorApp(root)
    root.mainloop()

    args = app.get_args()
    print("Selected Reduced Data Directory:", args["d"])
    print("Selected Output Directory:", args["o"])

    class Args: pass
    args_obj = Args()
    for key, val in args.items():
        setattr(args_obj, key, val)

    return args_obj


if __name__ == '__main__':
    # Set up argument parsing, but make the arguments optional (required=False)
    parser = argparse.ArgumentParser(description="PSF Photometry Tool")
    parser.add_argument('-d', '--data', type=str, required=False,
                       help="Directory containing reduced images")
    parser.add_argument('-o', '--output', type=str, required=False,
                       help="Output directory for results (Optional)")
    
    parsed_args = parser.parse_args()

    # Check if the user provided the data directory via the terminal
    if parsed_args.data:
        print("\nCommand-line arguments detected. Skipping UI...")
        # Create an args object to match the structure expected by the rest of the script
        class Args: pass
        args = Args()
        args.d = parsed_args.data
        # Default to the current working directory if no output flag is provided
        args.o = parsed_args.output if parsed_args.output else os.getcwd()
    else:
        # No command-line arguments provided, launch the directory selection UI
        args = main()
    
    print(f"\nRETRHO Data Reduction Pipeline Initiated...\nPerforming PSF Photometry on Data")

    # Get the path to the frame information dataframe and the uncertainties file
    frame_info_file = os.path.join(args.d, 'frame_info.csv')
    uncertainties_file = os.path.join(args.d, 'uncertainties.csv')

    # Read in the frame info and uncertainties
    frame_info_df = pd.read_csv(frame_info_file, sep=' ')
    uncertainties_df = pd.read_csv(uncertainties_file, sep=' ', names=['id', 'Value'], index_col=0)

    # Add the directory to the frame info dataframe
    frame_info_df['Directory'] = [args.d] * frame_info_df['File'].size
    
    # Verify all images are of the same object
    unique_objects = frame_info_df['Object'].unique()
    if len(unique_objects) > 1:
        print("Warning: Multiple objects found in the data directories:")
        for obj in unique_objects:
            print(f"  - {obj}")
        print("Please ensure all images are of the same object.")
        exit(1)

    # Verify all images are of the same object
    unique_objects = frame_info_df['Object'].unique()
    if len(unique_objects) > 1:
        print("Warning: Multiple objects found in the data directories:")
        for obj in unique_objects:
            print(f"  - {obj}")
        print("Please ensure all images are of the same object.")
        exit(1)

    # Get object name for file naming
    object_name = frame_info_df['Object'].iloc[0]

    # Extract the unique filters and display how many images per filter are there
    print(f"\n--- Image Summary for {object_name} ---")
    if 'Filter' in frame_info_df.columns:
        unique_filters = frame_info_df['Filter'].unique()
        filter_counts = frame_info_df['Filter'].value_counts()
        for filt, count in filter_counts.items():
            print(f"  - Filter {filt}: {count} images")
    else:
        print(f"  - Total images: {len(frame_info_df)}")
        unique_filters = ['None']

    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.o, 'PSF_Photometry_Results/')
    os.makedirs(output_dir, exist_ok=True)

    # Initialize reference variables before looping through filters
    reference_image = None
    reference_stars = None

    for filt in unique_filters:
        print(f"\n" + "="*50)
        print(f" PROCESSING FILTER: {filt}")
        print("="*50)
        
        # Subset the dataframe for the current filter
        if filt != 'None':
            filter_df = frame_info_df[frame_info_df['Filter'] == filt].copy()
        else:
            filter_df = frame_info_df.copy()

        # Define the number of bins (data points) from user input in the terminal
        total_images = len(filter_df)
        print(f"\n--- Image Binning Setup for Filter {filt} ---")
        print(f"Total number of images available: {total_images}")
        
        while True:
            try:
                user_input = input(f"Enter the number of bins you want to create for Filter {filt} (1 means all the images will be co-added, {total_images} means no binning).: ")
                num_bins = int(user_input)
                
                if not (1 <= num_bins <= total_images):
                    print(f"Please enter an integer between 1 and {total_images}")
                    continue
                    
                # Valid special cases: 1 bin (all images together) or num_bins == total_images (1 image per bin)
                if num_bins == 1 or num_bins == total_images:
                    break
                    
                # Check for forbidden cases where base images_per_bin is less than 2
                images_per_bin = total_images // num_bins
                if images_per_bin < 2:
                    print("Number of bins not allowed, all bins must have more than 1 image. Please choose a smaller bin number.")
                    continue
                    
                # If we get here, the input is valid
                break
                
            except ValueError:
                print("Invalid input. Please enter a numerical integer.")

        # Calculate how the images will be distributed among bins for the summary printout
        images_per_bin = total_images // num_bins
        remainder = total_images % num_bins

        print(f"\nCreating {num_bins} bin(s) for Filter {filt}...")
        if num_bins == total_images:
            print("  - Each bin will contain 1 image (No co-adding).")
        elif num_bins == 1:
            print(f"  - All {total_images} images will be co-added into 1 bin.")
        elif remainder == 0:
            print(f"  - Each bin will contain {images_per_bin} co-added images.")
        else:
            # Matches the distribution logic: remainder is spread across the first N bins
            print(f"  - The first {remainder} bin(s) will contain {images_per_bin + 1} co-added images.")
            print(f"  - The remaining {num_bins - remainder} bin(s) will contain {images_per_bin} co-added images.")
        print("-" * 30)

        # Run photometry for this specific filter subset
        psf_photometry = Compute_PSF_Photometry(
            filter_df, 
            uncertainties_df, 
            num_bins=num_bins,
            reference_image=reference_image,
            reference_stars=reference_stars
        )
        plt.show()

        # Capture the image and star positions from the FIRST filter to serve as a reference for the rest
        if reference_image is None and psf_photometry.star_positions:
            reference_image = psf_photometry.image_data
            reference_stars = psf_photometry.star_positions

        print(f"\nPhotometry complete for Filter {filt}.")

        # Save the results, injecting the filter name into the output file
        output_file = os.path.join(output_dir, f"{object_name}_{filt}_psf_photometry_{num_bins}_bins.csv")

        if psf_photometry.results_df is not None:
            results_df = psf_photometry.results_df
            results_df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_dir}")
        else:
            print(f"\nNo results to save for Filter {filt}. Star selection may have been cancelled.")