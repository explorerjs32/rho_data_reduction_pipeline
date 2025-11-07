import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle
from matplotlib.lines import Line2D
import matplotlib.widgets as widgets
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy import units as u
from astroquery.vizier import Vizier
from astropy.table import Table, QTable, vstack, conf, Column
from astropy.visualization import ImageNormalize, ZScaleInterval, LogStretch, LinearStretch
from photutils.detection import find_peaks
from photutils.aperture import CircularAperture, aperture_photometry
import glob
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

    #Defining global variable
    global wcs_file
    global wcs
    # Iterate through all directories
    for directory in directories:

        #Obtaining wcs file within directory
        wcs_file = os.path.join(directory, 'wcs.fits')

        #Acquiring wcs information
        wcs_file_header = fits.open(wcs_file)[0].header
        wcs = WCS(wcs_file_header)

        # Get all FITS files in the directory
        fits_files = [f for f in os.listdir(directory) if f.endswith('.fits') and f != 'wcs.fits']

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

class MedianImageSelector:
    """Class to select the image (filter) where the stars are the brightest."""

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
        self.image_data = None
        self.median_combined_images = {}
        self.selected_filter = None
        self.filtered_images_dict = {}
        self.other_frames_dict = {}
        self.median_combined_images = {}
        self.parse_filter_data()
        self.median_combine()
        
        #Designing the layout of the image display
        n_images = len(self.median_combined_images)
        n_cols = 2
        n_rows = math.ceil(n_images / n_cols)
        self.fig, self.axes = plt.subplots(n_rows, n_cols, figsize=(7, 4 * n_rows))
        self.fig.subplots_adjust(
            hspace=0.10,  # reduce vertical spacing between rows
            wspace=0.1,   # reduce horizontal spacing between columns
            bottom=0.15   # leave space at bottom for buttons
        )
        self.axes = self.axes.flatten()  # makes iteration easier
        
        self.display_images()
        self.create_widgets()
        self.selected_images = {}
        self.selected_filters = set()
        self.filter_names = list(self.median_combined_images.keys())
    
    def parse_filter_data(self):
        """Separates the light frames by filters for median combination."""

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

    def median_combine(self):
        """Combines images of the same filter using median combination."""

        # Median-combine images
        for filter, file_data in self.filtered_images_dict.items():
            self.median_combined_images[filter] = np.sum(file_data, axis=0)
            print(f"✅ Median-combination complete for {filter} filter.")

    def display_images(self):
            """Displays all median combined images."""

            # Add title and watermark
            self.fig.suptitle(f'Median-combined Image Selector', fontsize=14, fontweight='bold')
            self.fig.text(0.99, 0.01, 'RETRHO at UF', fontsize=10, fontweight='bold', ha='right', va='bottom', alpha=0.35)

            # Display median-combined image in each filter
            for ax, (filter_name, image_data) in zip(self.axes, self.median_combined_images.items()):
                norm = ImageNormalize(image_data, interval=ZScaleInterval())
                ax.imshow(image_data, origin='lower', cmap='gray', norm=norm)
                ax.set_title(filter_name)
                ax.axis('off')

            # Hide any unused subplots (if filters < total subplots)
            for ax in self.axes[len(self.median_combined_images):]:
                ax.axis('off')


    def create_widgets(self):
        """Creates widget containing the images and the buttons."""

        # Create a small axes for checkboxes
        check_ax = plt.axes([0.125, 0.08, 0.08, 0.1])  # adjust position as needed
        labels = list(self.median_combined_images.keys())
        visibility = [False] * len(labels)
        self.check = widgets.CheckButtons(check_ax, labels, visibility)
        self.check.on_clicked(self.on_check)
        #Define the done button to save image selection
        ax_done = plt.axes([0.25, 0.10, 0.1, 0.05])
        self.done_button = widgets.Button(ax_done, 'Done')
        self.done_button.on_clicked(self.save_selected_image)

    def on_check(self, label):
        """Handle checkbox selection."""

        if label in self.selected_filters:
            self.selected_filters.remove(label)
        else:
            self.selected_filters.add(label)
            

        # Highlight selections
        for ax, filter_name in zip(self.axes.flat, self.filter_names):
            color = 'yellow' if filter_name in self.selected_filters else 'black'
            lw = 3 if filter_name in self.selected_filters else 1
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(lw)

        self.fig.canvas.draw_idle()

    def save_selected_image(self, event):
        """Save the selected image."""

        if not self.selected_filters:
            print("No image selected!")
            return

        self.selected_images = {f: self.median_combined_images[f] for f in self.selected_filters}

        self.other_frames_dict = {f: self.median_combined_images[f] for f in self.median_combined_images if f not in self.selected_filters}

        print(self.other_frames_dict.keys())
        print(f"Stored {len(self.selected_images)} selected image(s) in memory.")
        plt.close(self.fig)



class AperturePhotometryTool:
    """Class to perform aperture photometry on the image that is selected from the MedianImageSelector class."""

    def __init__(self, selected_image):
        self.selected_image = selected_image # Using the first selected image for display  
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.subplots_adjust(bottom=0.25)  # Leaving space at the bottom for buttons
        self.display_image()
        self.create_widgets()
        self.current_index = 0
        self.current_xpeak, self.current_ypeak = -1, -1
        self.aperture_radius = 10.0
        self.apertures_dict = {}
        self.bg_apertures_dict = {}
        self.photometry_dict = {}

        self.star_text_box = []
        self.bg_text_box = []
        self.text_frame_num = None
        self.current_contour = None
        self.current_level = None
        self.current_vertices = None
        self.temp_contours = []
        self.contours_dict = {}
        self.astroObjects_set = set()
        self.bg_astroObjects_set = set()

        # Zoom in by using the scroll wheel
        self.scroll_cid = self.fig.canvas.mpl_connect('scroll_event', self.zoom_image)

        # Drag image by right-clicking
        self.dragging = False
        self.press_event = None
        self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def on_button_press(self, event):
        """Handle mouse button press events."""

        if event.button == 3:  # Right mouse button
            self.dragging = True
            self.press_event = event

    def on_button_release(self, event):
        """Handle mouse button release events."""

        if event.button == 3:  # Right mouse button
            self.dragging = False
            self.press_event = None

    def on_mouse_move(self, event):
        """Handle mouse movement events."""

        if self.dragging and self.press_event is not None:
            # Calculate the displacement
            dx = event.x - self.press_event.x
            dy = event.y - self.press_event.y
            
            # Control the speed of movement 
            speed_factor = 0.25
            dx *= speed_factor
            dy *= speed_factor
            
            # Get current limits
            x_lim = self.ax.get_xlim()
            y_lim = self.ax.get_ylim()

            # Calculate new limits
            new_x_lim = (x_lim[0] - dx, x_lim[1] - dx)

            # Invert dy for Y-Axis
            new_y_lim = (y_lim[0] - dy, y_lim[1] - dy)

            # Get image dimensions
            img_height = self.selected_image[list(self.selected_image.keys())[0]].shape[0]
            img_width = self.selected_image[list(self.selected_image.keys())[0]].shape[1]

            # Define edge threshold for limiting dragging
            edge_threshold = 10

            # Check horizontal limits to prevent dragging when close to the left or right edge
            if new_x_lim[0] < edge_threshold or new_x_lim[1] > (img_width - edge_threshold):
                # Prevent horizontal dragging
                new_x_lim = (x_lim[0], x_lim[1])  # Keep horizontal limits unchanged

            # Check vertical limits to prevent dragging when close to the top or bottom edge
            if new_y_lim[0] < edge_threshold or new_y_lim[1] > (img_height - edge_threshold):
                # Prevent vertical dragging
                new_y_lim = (y_lim[0], y_lim[1])  # Keep vertical limits unchanged

            # Set the limits
            self.ax.set_xlim(max(new_x_lim[0], 0), min(new_x_lim[1], img_width))
            self.ax.set_ylim(min(new_y_lim[0], img_height), max(new_y_lim[1], 0))  # Natural Y-axis orientation

            # Redraw the figure
            self.fig.canvas.draw_idle()

    def display_image(self):
        """Display the selected image."""

        # Add title and watermark
        self.fig.suptitle(f'Aperture Photometry Tool', fontsize=14, fontweight='bold')
        self.fig.text(0.99, 0.01, 'RETRHO at UF', fontsize=10, fontweight='bold', ha='right', va='bottom', alpha=0.35)

        # Display selected image
        for filter_name, image_data in self.selected_image.items():
            norm = ImageNormalize(image_data, interval=ZScaleInterval())
            img = self.ax.imshow(image_data, origin='lower', cmap='gray', norm=norm)
            self.ax.set_title(f'Filter {filter_name}', fontsize=14)
            self.ax.axis('off')
    
    def zoom_image(self, event):
            """Zoom in and out of the image using the scroll wheel."""

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
                image_width = self.selected_image[list(self.selected_image.keys())[0]].shape[1] 
                image_height = self.selected_image[list(self.selected_image.keys())[0]].shape[0]
                
                # Set the limits to the original image dimensions if zoomed out too much
                if new_width > image_width:
                    self.ax.set_xlim(0, image_width)
                    
                if new_height > image_height:
                    self.ax.set_ylim(0, image_height)
                    
                self.fig.canvas.draw_idle()

    def create_widgets(self):
        """Creates widget containing the image and the buttons."""

        # Define the position and size of the buttons
        ax_add_star =             plt.axes([0.185, 0.15, 0.15, 0.05])
        ax_increase_aper_radius = plt.axes([0.185, 0.09, 0.05, 0.05])
        ax_decrease_aper_radius = plt.axes([0.245, 0.09, 0.05, 0.05])
        ax_undo_star            = plt.axes([0.305, 0.09, 0.03, 0.05])

        ax_add_bg =               plt.axes([0.360, 0.15, 0.15, 0.05])
        ax_increase_bg_radius =   plt.axes([0.360, 0.09, 0.05, 0.05])
        ax_decrease_bg_radius =   plt.axes([0.420, 0.09, 0.05, 0.05])
        ax_undo_bg              = plt.axes([0.480, 0.09, 0.03, 0.05])

        ax_perform_phot =         plt.axes([0.635, 0.15, 0.18, 0.05])
        ax_done =                 plt.axes([0.715, 0.09, 0.10, 0.05])
  
        # Creating the buttons
        self.button_add_star = widgets.Button(ax_add_star, 'Add Star', color='LightGreen')
        self.button_increase_aper_radius = widgets.Button(ax_increase_aper_radius, '+')
        self.button_decrease_aper_radius = widgets.Button(ax_decrease_aper_radius, '–')
        self.button_undo_star            = widgets.Button(ax_undo_star, '↩', color="#FC6666FF")

        self.button_add_bg = widgets.Button(ax_add_bg, 'Add Background', color='LemonChiffon')
        self.button_increase_bg_radius = widgets.Button(ax_increase_bg_radius, '+')
        self.button_decrease_bg_radius = widgets.Button(ax_decrease_bg_radius, '–')
        self.button_undo_bg           = widgets.Button(ax_undo_bg, '↩', color='#FC6666FF')

        self.button_perform_phot = widgets.Button(ax_perform_phot, 'Aperture Photometry', color='lightblue')
        self.button_done = widgets.Button(ax_done, 'Done')

        # Assigning button functionalities
        self.button_add_star.on_clicked(lambda event: self.add_star_aperture(self.current_xpeak, self.current_ypeak))
        self.button_add_bg.on_clicked(lambda event: self.add_bg_aperture(self.current_xpeak, self.current_ypeak))
        self.button_increase_aper_radius.on_clicked(self.increase_aper_radius)
        self.button_decrease_aper_radius.on_clicked(self.decrease_aper_radius)
        self.button_increase_bg_radius.on_clicked(self.increase_bg_radius)
        self.button_decrease_bg_radius.on_clicked(self.decrease_bg_radius)
        self.button_perform_phot.on_clicked(lambda event: (self.perform_aperture_photometry(event)))
        self.button_done.on_clicked(self.done_button)

        self.button_undo_star.on_clicked(self.undo_star)
        self.button_undo_bg.on_clicked(self.undo_bg)
        

        # Create RectangleSelector for region selection
        self.rect_selector = widgets.RectangleSelector(self.ax, self.on_region_select, useblit=True, minspanx=5, minspany=5, spancoords='pixels', interactive=True, button=[1])
        
    def increase_aper_radius(self, event):
        current_index = self.current_index
        if current_index in self.apertures_dict and self.apertures_dict[current_index]:
            self.apertures_dict[current_index][-1]['radius'] += 1.0
            self.draw_star_apertures_for_current_image()

    def decrease_aper_radius(self, event):
        current_index = self.current_index
        if current_index in self.apertures_dict and self.apertures_dict[current_index]:
            self.apertures_dict[current_index][-1]['radius'] = max(1.0, self.apertures_dict[current_index][-1]['radius'] - 1.0)
            self.draw_star_apertures_for_current_image()    

    def increase_bg_radius(self, event):
        current_index = self.current_index
        if current_index in self.bg_apertures_dict and self.bg_apertures_dict[current_index]:
            self.bg_apertures_dict[current_index][-1]['radius'] += 1.0
            self.draw_bg_apertures_for_current_image()

    def decrease_bg_radius(self, event):
        current_index = self.current_index
        if current_index in self.bg_apertures_dict and self.bg_apertures_dict[current_index]:
            self.bg_apertures_dict[current_index][-1]['radius'] = max(1.0, self.bg_apertures_dict[current_index][-1]['radius'] - 1.0)
            self.draw_bg_apertures_for_current_image()    

    def done_button(self, event):
        """Closes the window when 'Done' is clicked."""

        if not self.astroObjects_set and not self.bg_astroObjects_set:
            print("❌ 'Done' pressed: Please, perform aperture photometry first.")
            return
        
        if not self.astroObjects_set:
            print("❌ 'Done' pressed: Please, perform aperture photometry first.")
            return
        
        if not self.bg_astroObjects_set:
            print("❌ 'Done' pressed: Please, perform aperture photometry first.")
            return

        if not self.photometry_dict:
            print("❌ 'Done' pressed: Please, perform aperture photometry first.")
            return
        
        # Number of objects selected in frame (number of stars + 1 background)
        number_of_objects_selected = len(self.astroObjects_set) + len(self.bg_astroObjects_set) 

        # Number of objects for which aperture photometry has been performed
        number_of_objects_phot_performed = len(self.photometry_dict[list(self.photometry_dict.keys())[0]])

        if number_of_objects_selected != number_of_objects_phot_performed:
            print("❌ 'Done' pressed: Please, perform aperture photometry first.")
            return

        filter_name = list(self.selected_image.keys())[0]
        
        print(f"✅✅✅ 'Done' pressed: Aperture photometry completed for {len(self.astroObjects_set)} stars.")
        print(list(self.photometry_dict.keys())[0])
        print(self.photometry_dict[f'Filter {filter_name}'])

        plt.close(self.fig)      

    def on_region_select(self, eclick, erelease):
        """Callback function for the RectangleSelector widget."""

        # Takes the value of the first key in the dictionary (the image)
        self.image_data = self.selected_image[list(self.selected_image.keys())[0]] 

        # Will remove temporary contours if they exist
        if getattr(self, 'temp_contours', None):  
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
    
    def add_star_aperture(self, xpeak, ypeak):
        current_index = self.current_index

        # Creates first instance of apertures_dict for the current image
        if current_index not in self.apertures_dict: 
            self.apertures_dict[current_index] = []

        objectNum = len(self.apertures_dict[self.current_index]) + 1

        self.apertures_dict[current_index].append({
            'center': (xpeak, ypeak),
            'radius': self.aperture_radius,  # current radius setting
            'objectNum': objectNum
        })

        self.draw_star_apertures_for_current_image()
        self.add_aperture_helper(objectNum)
        self.fig.canvas.draw_idle()
        print(f"🟢 'Add Star' pressed: Star {objectNum} added.")
        # print(len(self.star_text_box))
        # print(self.star_text_box)


    def undo_star(self, event):
        current_index = self.current_index

        if current_index in self.apertures_dict and self.apertures_dict[current_index]:
            # Remove the last added aperture
            self.apertures_dict[current_index].pop()  # This removes the last item
            self.astroObjects_set = set(list(self.astroObjects_set)[:-1])
            self.photometry_dict={}
            self.star_text_box[-1].remove()

            self.draw_star_apertures_for_current_image()  # Redraw the apertures
            self.fig.canvas.draw_idle()  # Update the canvas
            print("🔴 'Undo Star' pressed: Last star removed.")
        else:
            print("❌ 'Undo Star' pressed: No star to remove.")

    def undo_bg(self, event):
        current_index = self.current_index

        if current_index in self.bg_apertures_dict and self.bg_apertures_dict[current_index]:
            # Remove the last added aperture
            self.bg_apertures_dict[current_index].pop()  # This removes the last item
            self.bg_astroObjects_set = set(list(self.bg_astroObjects_set)[:-1])
            self.photometry_dict={}
            self.bg_text_box[-1].remove()

            self.draw_bg_apertures_for_current_image()  # Redraw the apertures
            self.fig.canvas.draw_idle()  # Update the canvas
            print("🔴 'Undo Background' pressed: Background removed.")
        else:
            print("❌ 'Undo Background' pressed: No background to remove.")

    def add_bg_aperture(self, xpeak, ypeak):
        current_index = self.current_index

        # Set maximum number of apertures
        max_apertures = 1  

        # Creates first instance for the current image
        if current_index not in self.bg_apertures_dict:  
            self.bg_apertures_dict[current_index] = []

        # Check if the maximum number of apertures has been reached
        if len(self.bg_apertures_dict[current_index]) >= max_apertures:
            print("❌ 'Add Background' pressed again: Only one background aperture needed.")
            return

        bg_objectNum = len(self.bg_apertures_dict[current_index]) + 1

        self.bg_apertures_dict[current_index].append({
            'center': (xpeak, ypeak),
            'radius': self.aperture_radius,  # current radius setting
            'bg_objectNum': bg_objectNum
        })

        self.draw_bg_apertures_for_current_image()
        self.add_bg_aperture_helper(bg_objectNum)
        self.fig.canvas.draw_idle()
        print(f"🟡 'Add Background' pressed: Background added.")

    def add_aperture_helper(self, objectNum):
        """
        Adds an aperture for the given object number.
        Updates astroObjects_set and apertures_dict accordingly.
        """

        if hasattr(self, 'astroObjects_set'):
            if objectNum not in self.astroObjects_set:
                self.astroObjects_set.add(objectNum)
            else:
                None

        num_of_images = len(self.selected_image)

        for num_of_image in range(num_of_images):
            if num_of_image not in self.apertures_dict:
                self.apertures_dict[num_of_image] = []
            if objectNum not in [a['objectNum'] for a in self.apertures_dict[num_of_image]]:
                self.apertures_dict[num_of_image].append({
                    'center': (self.current_xpeak, self.current_ypeak),
                    'radius': self.aperture_radius,
                    'objectNum': objectNum
                })
            else:
                None

    def add_bg_aperture_helper(self, bg_objectNum):
        """
        Adds an aperture for the given object number.
        Updates bg_astroObjects_set and bg_apertures_dict accordingly.
        """

        if hasattr(self, 'bg_astroObjects_set'):
            if bg_objectNum not in self.bg_astroObjects_set:
                self.bg_astroObjects_set.add(bg_objectNum)
            else:
                None

        num_of_images = len(self.selected_image)

        for num_of_image in range(num_of_images):
            if num_of_image not in self.bg_apertures_dict:
                self.bg_apertures_dict[num_of_image] = []
            if bg_objectNum not in [a['bg_objectNum'] for a in self.bg_apertures_dict[num_of_image]]:
                self.bg_apertures_dict[num_of_image].append({
                    'center': (self.current_xpeak, self.current_ypeak),
                    'radius': self.aperture_radius,
                    'objectNum': bg_objectNum
                })
            else:
                None

    def draw_star_apertures_for_current_image(self):
            # Remove old aperture patches
        if hasattr(self, 'temp_aperture_patches'):
            for patch in self.temp_aperture_patches:
                patch.remove()

        self.temp_aperture_patches = []

        self.star_text_box.clear() 

        # Draw apertures for current image
        current_index = self.current_index
        if current_index in self.apertures_dict:
            for aperture in self.apertures_dict[current_index]:
                x, y = aperture['center']
                r = aperture['radius']
                objectNum = aperture['objectNum']

                text_star = self.ax.text(x, y + 25, f'Star {objectNum}', color='lime', fontsize=10, ha='center', va='bottom', clip_on=True)
                self.star_text_box.append(text_star)        

                patch = Circle((x, y), r, edgecolor='lime', facecolor='none', lw=1)
                self.ax.add_patch(patch)
                self.temp_aperture_patches.append(patch)

                #self.ax.text(x, y + 25, f'Star {objectNum}', color='lime', fontsize=10, ha='center', va='bottom', clip_on=True)
        
            
        self.fig.canvas.draw_idle()
        
    def draw_bg_apertures_for_current_image(self):
            # Remove old aperture patches
        if hasattr(self, 'temp_bg_aperture_patches'):
            for patch in self.temp_bg_aperture_patches:
                patch.remove()

        self.temp_bg_aperture_patches = []

        # Draw apertures for current image
        current_index = self.current_index
        if current_index in self.bg_apertures_dict:
            for bg_aperture in self.bg_apertures_dict[current_index]:
                x, y = bg_aperture['center']
                r = bg_aperture['radius']

                text_bg = self.ax.text(x, y+25, f'Background', color='yellow', fontsize=10, ha='center', va='bottom', clip_on=True)
                self.bg_text_box.append(text_bg)

                patch = Circle((x, y), r, edgecolor='yellow', facecolor='none', lw=1)
                self.ax.add_patch(patch)
                self.temp_bg_aperture_patches.append(patch)

                #self.ax.text(x, y+25, f'Background', color='yellow', fontsize=10, ha='center', va='bottom', clip_on=True)

        self.fig.canvas.draw_idle()

    def perform_aperture_photometry(self, event):
        """Performs aperture photometry on all marked objects in the current image."""

        current_index = self.current_index

        if not self.astroObjects_set and not self.bg_astroObjects_set:
            print("❌ 'Aperture Photometry' pressed: Please, add at least one star aperture and one background aperture.")
            return

        if not self.astroObjects_set:
            print("❌ 'Aperture Photometry' pressed: Please, add at least one star aperture.")
            return
        
        if not self.bg_astroObjects_set:
            print("❌ 'Aperture Photometry' pressed: Please, add one background aperture.")
            return
        
        # Takes the value of the first key in the dictionary (the image)
        image_data = self.selected_image[list(self.selected_image.keys())[0]] 
        filter_name = list(self.selected_image.keys())[0]

        # Aperture photometry of background
        bg_position = [a['center'] for a in self.bg_apertures_dict[current_index]]
        bg_radius = [b['radius'] for b in self.bg_apertures_dict[current_index]]

        bg_photometry_table = pd.DataFrame({})

        bg_row=[]

        for i, (position, radius) in enumerate(zip(bg_position, bg_radius)):
            bg_aperture = [CircularAperture(position, radius)]
            
            bg_phot_table = aperture_photometry(image_data, bg_aperture)

            bg_row.append({
                'Star': f'Background',
                'X_Center': bg_phot_table['xcenter'][0],
                'Y_Center': bg_phot_table['ycenter'][0],
                'Radius': radius,
                'Aperture_Sum': bg_phot_table['aperture_sum_0'][0],
                'Aperture_Sum_Error': (np.abs(bg_phot_table['aperture_sum_0'][0]))**0.5
            })

        # Aperture photometry of stars
        positions = [a['center'] for a in self.apertures_dict[current_index]]
        radii = [b['radius'] for b in self.apertures_dict[current_index]]

        star_photometry_table = pd.DataFrame({})

        star_rows=[]

        for i, (position, radius) in enumerate(zip(positions, radii)):
            star_aperture = [CircularAperture(position, radius)]

            star_phot_table = aperture_photometry(image_data, star_aperture)

            # Uncertainty calculation
            star_error = (star_phot_table['aperture_sum_0'][0])**0.5
            bg_error = (np.abs(bg_phot_table['aperture_sum_0'][0]))**0.5    # Aperture sum may be negative
            aper_sum_error = (star_error**2 + bg_error**2)**0.5

            star_rows.append({
                'Star': f'Star {i+1}',
                'X_Center': star_phot_table['xcenter'][0],
                'Y_Center': star_phot_table['ycenter'][0],
                'Radius': radius, 
                'Aperture_Sum': (star_phot_table['aperture_sum_0'][0] - bg_phot_table['aperture_sum_0'][0]),
                'Aperture_Sum_Error': aper_sum_error
            })

        # Create the DataFrame
        star_photometry_table = pd.DataFrame(star_rows)
        bg_photometry_table = pd.DataFrame(bg_row)

        self.photometry_dict={f'Filter {filter_name}': pd.concat([star_photometry_table, bg_photometry_table], ignore_index=True)}
        
        print(f"🔵 'Aperture Photometry' pressed: Aperture photometry performed for {len(star_photometry_table)} stars.")
        print(list(self.photometry_dict.keys())[0])
        print(self.photometry_dict[f'Filter {filter_name}'])


if __name__ == '__main__':
    # Define the arguments to parse into the script
    parser = argparse.ArgumentParser(description="Arguments to parse for the PSF photometry pipeline. Primarily focusing on the directories where the data is stored.")
    
    parser.add_argument('-data', '--data', type=str, nargs='+', required=True, help="Single or multiple directories containing reduced images.")
    
    args = parser.parse_args()
    
    # Get the frame information from the reduced images
    median_frame_info_df = get_frame_info(args.data)
    
    #Initialize the MedianImageSelector class
    median_selected_images_class = MedianImageSelector(median_frame_info_df)
    plt.show()

    if median_selected_images_class.selected_images:
        aperture_photometry_class = AperturePhotometryTool(median_selected_images_class.selected_images)
        plt.show()


    # Get object name for file naming
    object_name = median_frame_info_df['Object'].iloc[0]
