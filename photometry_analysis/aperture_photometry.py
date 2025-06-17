import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.widgets as widgets
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ImageNormalize, ZScaleInterval, LogStretch, LinearStretch
from photutils.detection import find_peaks
import argparse
import os

import pdb

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
    reduced_frame_info = pd.DataFrame({'Directory': directories_list,
                                       'File': file_list,
                                       'Object': objects,
                                       'Date-Obs': dates,
                                       'Filter': filters,
                                       'Exptime': exposure_times})

    return reduced_frame_info


class aperturePhotometry:
    def __init__(self, frame_info_df):
        self.frame_info = frame_info_df
        self.median_frame_info = pd.DataFrame({ # Initializing empty DataFrame for median frame info
                    'Directory': [],
                    'File': [],
                    'Object': [],
                    'Date-Obs': [],
                    'Filter': []
                }) 
        self.current_index = 0
        self.image_data = None
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.text_box = None
        self.text_frame_num = None
        self.rect_selector = None
        self.current_contour = None
        self.current_level = None
        self.current_vertices = None
        self.temp_contours = []
        self.contours_dict = {}
        self.photometry_dict = {}
        self.filtered_images_dict = {}
        self.median_combined_images = {}
        self.parse_filter_data()
        self.median_combine()
        self.display_image()
        self.create_widgets()
        self.fig.canvas.mpl_connect('scroll_event', self.zoom_image)

    def parse_filter_data(self):
        """
        Separates the light frames by filters for median combination.
        """
        for index, row in self.frame_info.iterrows():
            directory = row['Directory']
            file = row['File']
            object = row['Object']
            date_obs = row['Date-Obs']
            filter_name = row['Filter']

            file_path = os.path.join(directory, file)

            # If the filter key doesn't exist in the dictionary, create an empty list for it
            if filter_name not in self.filtered_images_dict:
                self.filtered_images_dict[filter_name] = []
                # Appending the file information to the median_frame_info DataFrame
                new_row = pd.DataFrame([{
                    'Directory': directory,
                    'File': file,
                    'Object': object,
                    'Date-Obs': date_obs,
                    'Filter': filter_name
                }])
                self.median_frame_info = pd.concat([self.median_frame_info, new_row], ignore_index=True)

            # Add the file pixel data to the list corresponding to its filter
            try:
                with fits.open(file_path) as hdul:
                    file_data = hdul[0].data.astype(float)
                    self.filtered_images_dict[filter_name].append(file_data)
            except Exception as e:
                print(f"Error reading file {file} for filter {filter_name}: {e}")

        #pdb.set_trace()
        #print(self.median_frame_info)

        # print("Finished populating filtered_images_dict.")
        # for filter, files in self.filtered_images_dict.items():
        #     print(f"  {filter}: {len(files)} files")

    def median_combine(self):
        """
        Combines images of the same filter using median combination.
        """
        
        for filter, file_data in self.filtered_images_dict.items():
            self.median_combined_images[filter] = np.median(file_data, axis=0)

        #print(f"Median combination complete for {len(self.median_combined_images)} filters.")

    def display_image(self):
        """
        Displays the current image and the frame information.
        """
        # Clear the axis first
        self.ax.clear()
        
        # Set the current image_data to the median combined image for the current filter
        self.image_data = self.median_combined_images.get(self.median_frame_info['Filter'][self.current_index])
        
        # Display the image
        image_norm = ImageNormalize(self.image_data, interval=ZScaleInterval())
        self.ax.imshow(self.image_data, origin='lower', cmap='gray', norm=image_norm)
        
        # Create the text box with rounded edges
        textstr = (f"Median Combined Image for Filter: {self.median_frame_info['Filter'][self.current_index]}\n"
                f"Object: {self.median_frame_info['Object'][self.current_index]}\n"
                f"Date-Obs: {self.median_frame_info['Date-Obs'][self.current_index]}")
        #print(f"Displaying image: {self.median_frame_info['File'][self.current_index]} at index {self.current_index}")

        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        self.text_box = self.ax.text(0.02, 1.14, textstr, transform=self.ax.transAxes, fontsize=12,
                                    verticalalignment='top', bbox=props)

        # Add the frame number
        self.text_frame_num = self.ax.text(0.8, 1.05, f"Frame {self.current_index + 1}/{len(self.median_frame_info)}", 
                                        transform=self.ax.transAxes,
                                        fontsize=12, verticalalignment='top')
        
        self.ax.axis('off')
        
        # Display saved contours and labels if they exist for this image
        current_file = self.median_frame_info['File'][self.current_index]
        if self.current_index in self.contours_dict:
            for i, contour_info in enumerate(self.contours_dict[self.current_index], 1):
                x, y, width, height = contour_info['coords']
                vertices = contour_info['vertices']
                
                # Ensure the region is within image bounds
                if (x >= 0 and y >= 0 and 
                    x + width <= self.image_data.shape[1] and 
                    y + height <= self.image_data.shape[0]):
                    
                    # Create path from vertices and add as a patch
                    path = Path(vertices)
                    patch = PathPatch(path, facecolor='none', edgecolor='red', 
                                    linewidth=1, alpha=0.75)
                    self.ax.add_patch(patch)
                    
                    # Add star label if photometry data exists
                    if current_file in self.photometry_dict and i in self.photometry_dict[current_file]:
                        measurements = self.photometry_dict[current_file][i]
                        self.ax.text(measurements['xpeak'] - 25, measurements['ypeak'] + 25,
                                f'Star {i}', color='red', fontsize=10,
                                ha='center', va='bottom')
        
        self.fig.canvas.draw()

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

    def create_widgets(self):
        """
        Create the widgets used in the tool.
        """
        # Define the buttons to navigate between images
        ax_prev = plt.axes([0.125, 0.15, 0.1, 0.05])
        ax_next = plt.axes([0.25, 0.15, 0.1, 0.05])
        ax_add_star = plt.axes([0.375, 0.15, 0.1, 0.05])
        ax_perform_phot = plt.axes([0.5, 0.15, 0.15, 0.05])

        self.button_prev = widgets.Button(ax_prev, 'Previous')
        self.button_next = widgets.Button(ax_next, 'Next')
        self.button_add_star = widgets.Button(ax_add_star, 'Add Star')
        self.button_perform_phot = widgets.Button(ax_perform_phot, 'PSF Photometry')

        self.button_prev.on_clicked(self.prev_image)
        self.button_next.on_clicked(self.next_image)
        self.button_add_star.on_clicked(self.add_star)
        self.button_perform_phot.on_clicked(self.perform_photometry)
        
        # Create RectangleSelector for region selection
        self.rect_selector = widgets.RectangleSelector(self.ax, self.on_region_select, useblit=True,
                                                       minspanx=5, minspany=5,
                                                       spancoords='pixels', interactive=True)
        
    def on_region_select(self, eclick, erelease):
        """
        Callback function for the RectangleSelector widget.
        """
        # Get the coordinates of the selected region
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # Ensure the selected region is within the bounds of the image
        if width > 0 and height > 0 and x1 >= 0 and y1 >= 0 and x2 <= self.image_data.shape[1] and y2 <= self.image_data.shape[0]:
            self.display_psf(x1, y1, width, height)

    def display_psf(self, x, y, width, height):
        """
        Display temporary PSF contour on the current image while maintaining saved contours.
        """
        # Clear only temporary contours
        for contour in self.temp_contours:
            for coll in contour.collections:
                coll.remove()
        self.temp_contours = []
        
        # Extract the selected region
        sub_image = self.image_data[y:y+height, x:x+width]
        
        # Calculate stats for the selected region
        mean, median, std = sigma_clipped_stats(sub_image, sigma=3.0, maxiters=5)
        contour_level = mean + 10*std
        
        # Create a contour plot of the PSF
        contour = self.ax.contour(sub_image, levels=[contour_level], colors='red', 
                                linewidths=1, alpha=0.75, extent=(x, x+width, y, y+height))
        
        # Get the contour path vertices
        path = contour.collections[0].get_paths()[0]
        vertices = path.vertices
        
        # Store temporary contour and its data
        self.temp_contours.append(contour)
        self.current_contour = contour
        self.current_level = contour_level
        self.current_vertices = vertices
        
        # Redisplay saved contours for current image
        if self.current_index in self.contours_dict:
            for contour_info in self.contours_dict[self.current_index]:
                x_saved, y_saved, width_saved, height_saved = contour_info['coords']
                vertices_saved = contour_info['vertices']
                
                # Create path from saved vertices and add as a patch
                path = Path(vertices_saved)
                patch = PathPatch(path, facecolor='none', edgecolor='red', 
                                linewidth=1, alpha=0.75)
                self.ax.add_patch(patch)
        
        self.fig.canvas.draw_idle()

    def clear_temp_contours(self):
        """
        Clear temporary contours from the current image.
        """
        for contour in self.temp_contours:
            for coll in contour.collections:
                coll.remove()
        self.temp_contours = []
        self.current_contour = None

    def add_star(self, event):
        """
        Save the current contour information and initial photometry measurements.
        """
        if self.current_contour is not None and self.temp_contours:
            # Get the coordinates from the current contour's extent
            extent = self.current_contour.collections[0].get_paths()[0].get_extents()
            x = int(extent.x0)
            y = int(extent.y0)
            width = int(extent.x1 - extent.x0)
            height = int(extent.y1 - extent.y0)
            
            # Get current file name
            current_file = self.median_frame_info['File'][self.current_index]
            
            # Initialize the dictionaries if they don't exist
            if self.current_index not in self.contours_dict:
                self.contours_dict[self.current_index] = []
            if current_file not in self.photometry_dict:
                self.photometry_dict[current_file] = {}
            
            # Get the star number (1-based indexing)
            star_number = len(self.contours_dict[self.current_index]) + 1
            
            # Extract the region and find the peak
            region = self.image_data[y:y+height, x:x+width]
            max_counts = np.max(region)
            peak_y, peak_x = np.unravel_index(np.argmax(region), region.shape)
            
            # Calculate absolute peak position
            abs_peak_x = x + peak_x
            abs_peak_y = y + peak_y
            
            # Save contour information
            contour_info = {
                'coords': (x, y, width, height),
                'level': self.current_level,
                'vertices': self.current_vertices
            }
            self.contours_dict[self.current_index].append(contour_info)
            
            # Save initial photometry measurements
            self.photometry_dict[current_file][star_number] = {
                'xpeak': abs_peak_x,
                'ypeak': abs_peak_y,
                'peak_counts': max_counts
            }
            
            # Add star label
            self.ax.text(abs_peak_x - 25, abs_peak_y + 25, f'Star {star_number}', 
                        color='red', fontsize=10, ha='center', va='bottom')
            self.fig.canvas.draw_idle()

    def perform_photometry(self, event):
        """
        Perform PSF photometry on all marked stars in the current image.
        """
        current_file = self.median_frame_info['File'][self.current_index]
        
        if current_file in self.photometry_dict:
            for star_number, star_data in self.photometry_dict[current_file].items():
                # Get the contour information for this star
                contour_info = self.contours_dict[self.current_index][star_number - 1]
                x, y, width, height = contour_info['coords']
                
                # Extract the region
                region = self.image_data[y:y+height, x:x+width]
                
                # Calculate the sum flux (total counts within the contour)
                mask = Path(contour_info['vertices']).contains_points(
                    [(i, j) for i in range(x, x+width) for j in range(y, y+height)]
                ).reshape(height, width)
                
                sum_flux = np.sum(region[mask])
                sum_flux_err = np.sqrt(sum_flux)  # Assuming Poisson statistics
                
                # Update the photometry dictionary with flux measurements
                self.photometry_dict[current_file][star_number].update({
                    'sum_flux': sum_flux,
                    'sum_flux_err': sum_flux_err
                })
        
    def create_composite_dataframe(self):
        """
        Create a pandas DataFrame containing photometry measurements from all processed images.
        """
        data = []
        for file_name in self.photometry_dict.keys():
            # Get corresponding frame info
            median_frame_info_row = self.median_frame_info[self.median_frame_info['File'] == file_name].iloc[0]
            
            for star_number, measurements in self.photometry_dict[file_name].items():
                measurements_copy = measurements.copy()
                measurements_copy['star_number'] = star_number
                measurements_copy['file'] = file_name
                # Add additional information from median_frame_info
                measurements_copy['Date-Obs'] = median_frame_info_row['Date-Obs']
                measurements_copy['Filter'] = median_frame_info_row['Filter']
                data.append(measurements_copy)
        
        if data:
            df = pd.DataFrame(data)
            # Reorder columns to put file, date, filter, and star_number first
            cols = ['file', 'Date-Obs', 'Filter', 'star_number'] + \
                [col for col in df.columns if col not in ['file', 'Date-Obs', 'Filter', 'star_number']]
            df = df[cols]
            return df
        return None

    def next_image(self, event):
        """
        Move to the next image in the list.
        """
        if self.current_index < len(self.median_frame_info) - 1:
            # Display composite photometry data including all processed images
            composite_df = self.create_composite_dataframe()
            if composite_df is not None:
                print("\nComposite Photometry Measurements:")
                print(composite_df.to_string(index=False))
                print("\n" + "="*50 + "\n")  # Separator line
            
            self.clear_temp_contours()  # Clear temporary contours
            self.current_index += 1
            self.display_image()
            self.update_button_status()
            self.ax.set_xlim(0, self.image_data.shape[1])
            self.ax.set_ylim(0, self.image_data.shape[0])
            self.fig.canvas.draw_idle()

    def prev_image(self, event):
        """
        Move to the previous image in the list.
        """
        if self.current_index > 0:
            self.clear_temp_contours()  # Clear temporary contours
            self.current_index -= 1
            self.display_image()
            self.update_button_status()
            self.ax.set_xlim(0, self.image_data.shape[1])
            self.ax.set_ylim(0, self.image_data.shape[0])
            self.fig.canvas.draw_idle()

    def update_button_status(self):
        """
        Update the button status based on the current image index.
        """
        if self.current_index == 0:
            self.button_prev.set_active(False)
            
        else:
            self.button_prev.set_active(True)
        
        if self.current_index == len(self.median_frame_info) - 1:
            self.button_next.set_active(False)
            
        else:
            self.button_next.set_active(True)


if __name__ == '__main__':
    # Define the arguments to parse into the script
    parser = argparse.ArgumentParser(description="Arguments to parse for the PSF photometry pipeline. Primarily focusing on the directories where the data is stored.")
    
    parser.add_argument('-data', '--data', type=str, nargs='+', required=True, help="Single or multiple directories containing reduced images.")
    
    args = parser.parse_args()
    
    # Get the frame inofrmation from the reduced images
    median_frame_info_df = get_frame_info(args.data)
    
    # Initialize the aperturePhotometry class
    aperture_photometry = aperturePhotometry(median_frame_info_df)
    
    plt.show()