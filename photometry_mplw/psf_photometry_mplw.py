import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ImageNormalize, ZScaleInterval, LogStretch, LinearStretch
from photutils.detection import find_peaks
import argparse
import os


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


class PSFPhotometry:
    def __init__(self, frame_info_df):
        self.frame_info = frame_info_df
        self.current_index = 0
        self.image_data = None
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.text_box = None
        self.text_frame_num = None
        self.psf_results = {}
        self.rect_selector = None
        self.current_contour = None
        self.contours_dict = {}
        self.display_image()
        self.create_widgets()
        self.fig.canvas.mpl_connect('scroll_event', self.zoom_image)

    def display_image(self):
        """
        Displays the current image and the frame information.
        """
        # Read the current FITS file
        file_path = os.path.join(self.frame_info['Directory'][self.current_index],
                                 self.frame_info['File'][self.current_index])
        
        self.image_data = fits.getdata(file_path)
        
        # Display the image
        image_norm = ImageNormalize(self.image_data, interval=ZScaleInterval())
        self.ax.imshow(self.image_data, origin='lower', cmap='gray', norm=image_norm)
        
        # Remove the previous text box
        if self.text_box:
            self.text_box.remove()

        if self.text_frame_num:
            self.text_frame_num.remove()
            
        # Create the text box with rounded edges
        textstr = (f"File: {self.frame_info['File'][self.current_index]}\n"
                   f"Object: {self.frame_info['Object'][self.current_index]}\n"
                   f"Filter: {self.frame_info['Filter'][self.current_index]}\n"
                   f"Exposure Time: {self.frame_info['Exptime'][self.current_index]} s")

        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        self.text_box = self.ax.text(0.02, 1.2, textstr, transform=self.ax.transAxes, fontsize=12,
                                     verticalalignment='top', bbox=props)

        # Add the frame number out of the total number of frames
        self.text_frame_num = self.ax.text(0.8, 1.05, f"Frame {self.current_index + 1}/{len(self.frame_info)}", transform=self.ax.transAxes,
                                           fontsize=12, verticalalignment='top')
        
        self.ax.axis('off')
        
        # Display PSF information if available
        if self.current_index in self.psf_results:
            self.display_psf_info()
        
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
        
        self.button_prev = widgets.Button(ax_prev, 'Previous')
        self.button_next = widgets.Button(ax_next, 'Next')
        self.button_prev.on_clicked(self.prev_image)
        self.button_next.on_clicked(self.next_image)
        
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
            # Perform PSF photometry
            self.perform_psf_photometry(x1, y1, width, height)
        else:
            pass

    def display_psf_info(self):
        """
        Display PSF information for previously analyzed images.
        """
        if self.current_index in self.psf_results:
            df = self.psf_results[self.current_index]
            # Iterate through all peaks in the dataframe
            for i in range(len(df)):
                x = df['x_peak'].iloc[i]
                y = df['y_peak'].iloc[i]
                width = height = 20  # You can adjust this size
                self.display_psf(int(x-width/2), int(y-height/2), width, height)

        
        self.fig.canvas.draw_idle()

    def perform_psf_photometry(self, x, y, width, height):
        """
          Perform PSF photometry on the selected region.
          """
        # Create a dataframe to store the results
        df = pd.DataFrame(columns=['File', 'Object', 'Exptime', 'Filter', 'Star', 'X', 'Y', 'Flux', 'Error'])
        
        # Fill in the file, object, exptime, and filter information
        df['File'] = [self.frame_info['File'][self.current_index]]
        df['Object'] = [self.frame_info['Object'][self.current_index]]
        df['Exptime'] = [self.frame_info['Exptime'][self.current_index]]
        df['Filter'] = [self.frame_info['Filter'][self.current_index]]
        
        # Extract the selected region
        sub_image = self.image_data[y:y+height, x:x+width]
        
        # Perform PSF photometry on the selected region
        mean, median, std = sigma_clipped_stats(sub_image, sigma=3.0, maxiters=5)
        
        # Find the peaks in the selected region
        peaks_tbl = find_peaks(sub_image, mean + 5 * std, box_size=5).to_pandas().sort_values(by='peak_value', ascending=False).reset_index(drop=True)
        
        # Create a cutout region of the star
        star_cutout = sub_image[int(peaks_tbl['y_peak'].iloc[0]) - height:int(peaks_tbl['y_peak'].iloc[0]) + height,
                                int(peaks_tbl['x_peak'].iloc[0]) - width:int(peaks_tbl['x_peak'].iloc[0]) + width]
        
        # Calculate stats for the star cutout
        star_mean, star_median, star_std = sigma_clipped_stats(star_cutout, sigma=3.0, maxiters=5)
        
        # Calculate the flux and error
        flux = star_cutout[star_cutout > 3.*star_std].sum()
        flux_err = np.sqrt(flux)
        
        # Fill in the PSF photometry results
        df['Star'] = [1]
        df['x_peak'] = [x + peaks_tbl['x_peak'].iloc[0]]
        df['y_peak'] = [y + peaks_tbl['y_peak'].iloc[0]]
        df['Flux'] = [flux]
        df['Error'] = [flux_err]
        
        # Store the dataframe in the dictionary
        self.psf_results[self.current_index] = df
        
        # Display PSF information on the image
        self.display_psf(x, y, width, height)

    def display_psf(self, x, y, width, height):
        """
        Display the PSF photometry information on the image as a contour.
        """
        
        # Extract the selected region
        sub_image = self.image_data[y:y+height, x:x+width]
        
        # Calculate stats for the selected region
        mean, median, std = sigma_clipped_stats(sub_image, sigma=3.0, maxiters=5)
        
        # Create a contour plot of the PSF
        contour = self.ax.contour(sub_image, levels=[mean + 3*std], colors='red', linewidths=1, alpha=0.75, extent=(x, x+width, y, y+height))

        # Initialize list for current image if not exists
        if self.current_index not in self.contours_dict:
            self.contours_dict[self.current_index] = []

        # Add new contour to the list for current image
        self.contours_dict[self.current_index].append(contour)
        self.current_contour = contour
    
        self.fig.canvas.draw_idle()

        print(self.current_index,self.contours_dict)

    def clear_contour(self):
        """
        Clear all PSF contours from the current image.
        """
        if self.current_index in self.contours_dict:
            for contour in self.contours_dict[self.current_index]:
                for coll in contour.collections:
                    coll.remove()
            self.contours_dict[self.current_index] = []
        self.current_contour = None
        
    def next_image(self, event):
        """
        Move to the next image in the list.
        """
        # if the current index is less than the total number of frames, move to the next image
        if self.current_index < len(self.frame_info) - 1:
            self.clear_contour()
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
        # If the current index is greater than 0, move to the previous image
        if self.current_index > 0:
            self.clear_contour()
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
        
        if self.current_index == len(self.frame_info) - 1:
            self.button_next.set_active(False)
            
        else:
            self.button_next.set_active(True)


if __name__ == '__main__':
    # Define the arguments to parse into the script
    parser = argparse.ArgumentParser(description="Arguments to parse for the PSF photometry pipeline. Primarily focusing on the directories where the data is stored.")
    
    parser.add_argument('-data', '--data', type=str, nargs='+', required=True, help="Single or multiple directories containing reduced images.")
    
    args = parser.parse_args()
    
    # Get the frame inofrmation from the reduced images
    frame_info_df = get_frame_info(args.data)
    
    # Initialize the PSFPhotometry class
    psf_photometry = PSFPhotometry(frame_info_df)
    
    plt.show()