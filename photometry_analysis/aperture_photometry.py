import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle
from matplotlib.lines import Line2D
import matplotlib.widgets as widgets
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ImageNormalize, ZScaleInterval, LogStretch, LinearStretch
from photutils.detection import find_peaks
from photutils.aperture import CircularAperture, aperture_photometry
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
                    'Filter': [],
                    'Exptime': []
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
        self.apertures_dict = {}
        #self.temp_aperture_patches = []  # Store temporary aperture patches
        self.astroObjects_set = set()
        self.current_xpeak, self.current_ypeak = -1, -1
        self.aperture_radius = 10.0
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
            exptime = row['Exptime']

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
                    'Filter': filter_name,
                    'Exptime': exptime
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
        self.image_data = self.median_combined_images.get(self.median_frame_info['Filter'][self.current_index]) # self.current_index is initially 0
        
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
        ax_add_object = plt.axes([0.375, 0.15, 0.1, 0.05])
        ax_perform_phot = plt.axes([0.5, 0.15, 0.15, 0.05])

        self.button_prev = widgets.Button(ax_prev, 'Previous')
        self.button_next = widgets.Button(ax_next, 'Next')
        self.button_add_object = widgets.Button(ax_add_object, 'Add Object')
        self.button_perform_phot = widgets.Button(ax_perform_phot, 'Aperture Photometry')

        self.button_prev.on_clicked(self.prev_image)
        self.button_next.on_clicked(self.next_image)
        self.button_add_object.on_clicked(lambda event: self.add_aperture(self.current_xpeak, self.current_ypeak))
        self.button_perform_phot.on_clicked(self.perform_aperture_photometry)
        
        # Add axes for radius adjustment buttons
        ax_increase_radius = plt.axes([0.675, 0.15, 0.05, 0.05])
        ax_decrease_radius = plt.axes([0.75, 0.15, 0.05, 0.05])

        self.button_increase_radius = widgets.Button(ax_increase_radius, '+')
        self.button_decrease_radius = widgets.Button(ax_decrease_radius, '–')

        self.button_increase_radius.on_clicked(self.increase_radius)
        self.button_decrease_radius.on_clicked(self.decrease_radius)

        # Create RectangleSelector for region selection
        self.rect_selector = widgets.RectangleSelector(self.ax, self.on_region_select, useblit=True,
                                                       minspanx=5, minspany=5,
                                                       spancoords='pixels', interactive=True)
        
    def on_region_select(self, eclick, erelease):
        """
        Callback function for the RectangleSelector widget.
        """
        if getattr(self, 'temp_contours', None):  # Will remove temporary contours if they exist
            for line in self.temp_contours:
                line.remove()
        self.temp_contours = []

        # Get the coordinates of the selected region
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # Ensure the selected region is within the bounds of the image
        if width > 0 and height > 0 and x1 >= 0 and y1 >= 0 and x2 <= self.image_data.shape[1] and y2 <= self.image_data.shape[0]:
            self.current_xpeak, self.current_ypeak = self.display_object_peak(x1, y1, width, height)

    def display_object_peak(self, x, y, width, height):
        """
        Find the peak brightness pixel in a selected region of the image.
        Returns (xpeak, ypeak) in full image coordinates.
        Draws a cross at the peak location.
        """
        sub_image = self.image_data[y:y + height, x:x + width]

        # Find the peak location in the sub-image
        ypeak_local, xpeak_local = np.unravel_index(np.argmax(sub_image), sub_image.shape)

        # Convert to image coordinates
        xpeak = x + xpeak_local
        ypeak = y + ypeak_local
        #print(f"Peak found at (x, y): ({xpeak}, {ypeak}) with counts: {sub_image[ypeak_sub, xpeak_sub]}")
        
        line_size = 10
        # Draw a cross at the peak location
        line1 = Line2D([xpeak - line_size, xpeak + line_size], 
                       [ypeak - line_size, ypeak + line_size], color='red', lw=0.8)
        line2 = Line2D([xpeak - line_size, xpeak + line_size], 
                       [ypeak + line_size, ypeak - line_size], color='red', lw=0.8)
        self.ax.add_line(line1)
        self.ax.add_line(line2)
        self.temp_contours = [line1, line2]
        self.fig.canvas.draw_idle()
        return xpeak, ypeak
    
    def add_aperture(self, xpeak, ypeak):
        current_index = self.current_index

        if current_index not in self.apertures_dict: # Creates first instance of apertures_dict for the current image
            self.apertures_dict[current_index] = []

        objectNum = len(self.apertures_dict[self.current_index]) + 1

        self.apertures_dict[current_index].append({
            'center': (xpeak, ypeak),
            'radius': self.aperture_radius,  # current radius setting
            'objectNum': objectNum
        })

        self.draw_apertures_for_current_image()
        self.add_aperture_helper(objectNum)
        self.fig.canvas.draw_idle()

    def add_aperture_helper(self, objectNum):
        """
        Adds an aperture for the given object number across all median combined images.
        Updates astroObjects_set and apertures_dict accordingly.
        """

        if hasattr(self, 'astroObjects_set'):
            if objectNum not in self.astroObjects_set:
                self.astroObjects_set.add(objectNum)
                #print(f"Added Star {objectNum} to astroObjects_set.")
            else:
                None
                #print(f"Star {objectNum} already exists in astroObjects_set.")

        num_of_images = len(self.median_combined_images)
        #print(f'Num of Images: {num_of_images} \n')

        for num_of_image in range(num_of_images):
            if num_of_image not in self.apertures_dict:
                self.apertures_dict[num_of_image] = []
            if objectNum not in [a['objectNum'] for a in self.apertures_dict[num_of_image]]:
                self.apertures_dict[num_of_image].append({
                    'center': (self.current_xpeak, self.current_ypeak),
                    'radius': self.aperture_radius,
                    'objectNum': objectNum
                })
                #print(f"Added Star {objectNum} to image {num_of_image} apertures.")
            else:
                None
                #print(f"Star {objectNum} already exists in image {num_of_image} apertures.")


    def draw_apertures_for_current_image(self):
            # Remove old aperture patches
        if hasattr(self, 'temp_aperture_patches'):
            for patch in self.temp_aperture_patches:
                patch.remove()
        self.temp_aperture_patches = []
        # Draw apertures for current image
        current_index = self.current_index
        if current_index in self.apertures_dict:
            for aperture in self.apertures_dict[current_index]:
                x, y = aperture['center']
                r = aperture['radius']
                objectNum = aperture['objectNum']
                patch = Circle((x, y), r, edgecolor='lime', facecolor='none', lw=1)
                self.ax.add_patch(patch)
                self.temp_aperture_patches.append(patch)
                self.ax.text(x - 25, y + 25, f'Star {objectNum}', 
                color='lime', fontsize=10, ha='center', va='bottom', clip_on=True)

        self.fig.canvas.draw_idle()

    def display_psf(self, x, y, width, height):
        """

                                 Soon to be removed.

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
        if hasattr(self, 'temp_contours'):
            for line in self.temp_contours:
                line.remove()
            self.temp_contours = []
        self.current_contour = None

    def add_star(self, event):
        """

                                Soon to be removed.

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

        # Update other images with the same star number and peak positions
        if current_file in self.photometry_dict and star_number in self.photometry_dict[current_file]:
                peak_data = self.photometry_dict[current_file][star_number]

                for file_name in self.median_frame_info['File']:
                    if file_name == current_file:
                        continue

                    if file_name not in self.photometry_dict:
                        self.photometry_dict[file_name] = {}

                    if star_number not in self.photometry_dict[file_name]:
                        self.photometry_dict[file_name][star_number] = {}

                    self.photometry_dict[file_name][star_number] = {
                        'xpeak': peak_data['xpeak'],
                        'ypeak': peak_data['ypeak']}

    def perform_aperture_photometry(self, event):
        """
        Perform aperture photometry on all marked objects in current image
        """
        current_index = self.current_index

        if current_index not in self.apertures_dict:
            print("No apertures defined for this image.")
            return
        
        image_data = self.image_data
        positions = [a['center'] for a in self.apertures_dict[current_index]]
        radii = [a['radius'] for a in self.apertures_dict[current_index]]

        current_filter = self.median_frame_info['Filter'][self.current_index]
        exptime = self.median_frame_info['Exptime'][self.current_index] #pulling the exptime for instr_mag calculations

        print(f"\nPerforming aperture photometry on filter: {current_filter}\n  Positions (px): {positions}\n  Radii: {radii}\n")
        
        if len(set(radii)) == 1: # if the list of radii has only one unique value, pass it as a single value
            aperture = CircularAperture(positions, r=radii[0])
        else:
            aperture = [CircularAperture(positions, r) for r in radii]

        phot_table = aperture_photometry(image_data, aperture)
        aperture_sum_err = (phot_table['aperture_sum'].data)**0.5
        phot_table['aperture_sum_err'] = aperture_sum_err

        #Calculate the instrumental magnitudes
        star_countrate = phot_table['aperture_sum'].data / exptime
        instr_mag = -2.5 * np.log10(star_countrate)
        phot_table['instrumental_magnitudes'] = instr_mag

        print(phot_table)


    def perform_photometry(self, event):
        """

                                Soon to be removed.

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

                                Needs to be updated.

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
            self.draw_apertures_for_current_image()
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
            self.draw_apertures_for_current_image()
            self.fig.canvas.draw_idle()

    def update_button_status(self):
        """
        Update the button status based on the current image index.
        """
        if self.current_index == 0:
            self.button_prev.set_active(False)
            
        else:
            self.button_prev.set_active(True)
        
        if self.current_index == len(self.median_combined_images) - 1:
            self.button_next.set_active(False)
            
        else:
            self.button_next.set_active(True)

    def increase_radius(self, event):
        current_index = self.current_index
        if current_index in self.apertures_dict and self.apertures_dict[current_index]:
            self.apertures_dict[current_index][-1]['radius'] += 1.0
            self.draw_apertures_for_current_image()

    def decrease_radius(self, event):
        current_index = self.current_index
        if current_index in self.apertures_dict and self.apertures_dict[current_index]:
            self.apertures_dict[current_index][-1]['radius'] = max(1.0, self.apertures_dict[current_index][-1]['radius'] - 1.0)
            self.draw_apertures_for_current_image()

if __name__ == '__main__':
    # Define the arguments to parse into the script
    parser = argparse.ArgumentParser(description="Arguments to parse for the PSF photometry pipeline. Primarily focusing on the directories where the data is stored.")
    
    parser.add_argument('-data', '--data', type=str, nargs='+', required=True, help="Single or multiple directories containing reduced images.")
    
    args = parser.parse_args()
    
    # Get the frame inofrmation from the reduced images
    median_frame_info_df = get_frame_info(args.data)
    
    # Initialize the aperturePhotometry class
    aperture_photometry_class = aperturePhotometry(median_frame_info_df)
    
    plt.show()