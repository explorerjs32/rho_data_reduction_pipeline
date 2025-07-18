import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.widgets import Button, RectangleSelector
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ImageNormalize, ZScaleInterval
import argparse
import os
from tqdm import tqdm
from scipy import ndimage


class PSFPhotometry:
    def __init__(self, frame_info_df, uncertainties_df, gain=0.37):
        self.frame_info = frame_info_df
        self.uncertainties_df = uncertainties_df
        self.gain = 0.37
        self.images = {}
        self.load_all_images()
        
        self.star_positions = {}  # {star_number: (x, y)}
        self.photometry = {}     # {filename: {star_number: flux}}
        
        self.current_star = 1
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.setup_plot()
        self.create_widgets()
        
        # Connect zoom event
        self.scroll_cid = self.fig.canvas.mpl_connect('scroll_event', self.zoom_image)

    def zoom_image(self, event):
        """
        Zoom in and out of the image using the scroll wheel.
        """
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            scale_factor = 1.25 if event.button == 'down' else 0.75
            
            # Get current x and y limits
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            
            # Calculate the new limits
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            
            xcenter = (cur_xlim[0] + cur_xlim[1]) / 2
            ycenter = (cur_ylim[0] + cur_ylim[1]) / 2
            
            self.ax.set_xlim([xcenter - new_width / 2, xcenter + new_width / 2])
            self.ax.set_ylim([ycenter - new_height / 2, ycenter + new_height / 2])
            
            # Get the original image dimensions
            image_width = self.image_data.shape[1]
            image_height = self.image_data.shape[0]
            
            # Set the limits to the original image dimensions if zoomed out too much
            if new_width > image_width:
                self.ax.set_xlim(0, image_width)
                
            if new_height > image_height:
                self.ax.set_ylim(0, image_height)
                
            self.fig.canvas.draw_idle()

    def load_all_images(self):
        """Load all images into memory."""
        for _, row in self.frame_info.iterrows():
            filepath = os.path.join(row['Directory'], row['File'])
            self.images[row['File']] = fits.getdata(filepath)

    def setup_plot(self):
        """Initialize the plot with the first image."""
        first_file = self.frame_info['File'].iloc[0]
        self.image_data = self.images[first_file]
        
        # Display the image
        image_norm = ImageNormalize(self.image_data, interval=ZScaleInterval())
        self.ax.imshow(self.image_data, origin='lower', cmap='gray', norm=image_norm)
        
        # Add image info text box
        textstr = (f"File: {first_file}\n"
                  f"Object: {self.frame_info['Object'].iloc[0]}\n"
                  f"Exposure Time: {self.frame_info['Exptime'].iloc[0]} secs\n"
                  f"Filter: {self.frame_info['Filter'].iloc[0]}")
        self.ax.text(0.02, 1.11, textstr, transform=self.ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='top')

        self.ax.axis('off')

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
        self.done_button_ax = plt.axes([0.55, 0.02, 0.25, 0.04])
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
        self.compute_all_photometry()

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
                
                # If center is not in bright region, find nearest one
                if center_region == 0:
                    y_grid, x_grid = np.ogrid[:star_cutout.shape[0], :star_cutout.shape[1]]
                    distance = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
                    bright_positions = np.where(bright_pixels)
                    
                    if len(bright_positions[0]) > 0:
                        dist_to_bright = distance[bright_positions]
                        closest_idx = np.argmin(dist_to_bright)
                        center_region = labeled_regions[bright_positions[0][closest_idx],
                                                    bright_positions[1][closest_idx]]
                
                # Create mask for central region
                final_mask = labeled_regions == center_region
                
                # Calculate flux and store results
                flux_out = np.sum(star_cutout[final_mask]) * self.gain
                self.photometry[filename][f"Flux_Star_{star_num}"] = flux_out
                
                # Calculate uncertainties
                read_noise = self.uncertainties_df.loc['Read_Noise', 'Values']
                dark_current = self.uncertainties_df.loc[f'Dark_Current_{exptime}s', 'Values']
                flat_noise = self.uncertainties_df.loc[f'Flat_{filter_}_Noise', 'Values']
                
                # Calculate flux uncertainty
                flux_noise = np.sqrt(
                    flux_out +  # Shot noise
                    (final_mask.sum() * dark_current * self.gain * exptime) +  # Dark current noise
                    (flat_noise * self.gain) +  # Flat field noise
                    (read_noise * self.gain)**2  # Read noise
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
            row.update(self.photometry[filename])
            data.append(row)
        
        results_df = pd.DataFrame(data)
        print("\nPhotometry Results:")
        print(results_df)
        
        return results_df


def get_frame_info(directory):
    """
    Extracts information from FITS file headers for reduced frames.
    """
    directories_list, file_list = [], []
    objects, dates, filters, exposure_times = [], [], [], []

    fits_files = [f for f in os.listdir(directory) if f.endswith('.fits')]

    for file in fits_files:
        try:
            header = fits.getheader(os.path.join(directory, file))
            directories_list.append(directory)
            file_list.append(file)
            objects.append(header.get('OBJECT', 'Unknown'))
            dates.append(header.get('DATE-OBS', 'Unknown'))
            filters.append(header.get('FILTER', 'Unknown'))
            exposure_times.append(header.get('EXPTIME', 'Unknown'))
        except Exception as e:
            print(f"Error processing file {file} in {directory}: {e}")
            continue

    return pd.DataFrame({
        'Directory': directories_list,
        'File': file_list,
        'Object': objects,
        'Date-Obs': dates,
        'Filter': filters,
        'Exptime': exposure_times
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PSF Photometry Tool")
    parser.add_argument('-d', '--data', type=str, required=True,
                       help="Directories containing reduced images")
    parser.add_argument('-u', '--uncertainties', type=str, required=True,
                       help="Path to uncertainties CSV file")
    args = parser.parse_args()
    
    # Read frame info and uncertainties
    frame_info_df = get_frame_info(args.data)
    uncertainties_df = pd.read_csv(args.uncertainties, sep=' ', names=['id','Values'], index_col=0)

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

    # Run photometry
    psf_photometry = PSFPhotometry(frame_info_df, uncertainties_df)
    plt.show()

    # After window is closed, results are computed automatically
    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.data, 'PSF_Photometry_Results/')
    os.makedirs(output_dir, exist_ok=True)

    # Save the results
    output_file = os.path.join(output_dir, f"{object_name}_psf_photometry.csv")
    results_df = psf_photometry.compute_all_photometry()
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")