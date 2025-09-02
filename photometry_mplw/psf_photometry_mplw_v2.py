import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.widgets as widgets
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.stats import SigmaClip
from astropy.visualization import ImageNormalize, ZScaleInterval
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_threshold, detect_sources
from photutils.utils import circular_footprint
import argparse
import os
from tqdm import tqdm

class PSFPhotometry:
    def __init__(self, frame_info_df):
        self.frame_info = frame_info_df
        self.images = {}
        self.load_all_images()
        
        self.star_positions = {}  # {star_number: (x, y)}
        self.psf_contours = {}   # {filename: {star_number: contour_vertices}}
        self.photometry = {}     # {filename: {star_number: flux}}
        self.noise = {}          # {filename: {star_number: noise}}
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
                  f"Filter: {self.frame_info['Filter'].iloc[0]}\n"
                #   f"Dark Current: {self.frame_info['Dark Current'].iloc[0]}"
                  )
        self.ax.text(0.02, 1.11, textstr, transform=self.ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='top')

        self.ax.axis('off')

    def create_widgets(self):
        """Create the interface widgets."""
        # Rectangle selector for star selection
        self.rect_selector = widgets.RectangleSelector(
            self.ax, self.on_select,
            useblit=True,
            props=dict(facecolor='red', edgecolor='red', alpha=0.2),
            interactive=True
        )
        
        # Draw PSF button
        self.draw_psf_button_ax = plt.axes([0.35, 0.02, 0.15, 0.04])
        self.draw_psf_button = widgets.Button(self.draw_psf_button_ax, 'Draw PSF')
        self.draw_psf_button.on_clicked(self.draw_all_psf)
        
        # Done button
        self.done_button_ax = plt.axes([0.55, 0.02, 0.25, 0.04])
        self.done_button = widgets.Button(self.done_button_ax, 'Done with Star Selection')
        self.done_button.on_clicked(self.finish_selection)
        
        # Initially disable buttons
        self.draw_psf_button.set_active(False)
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
        
        # Enable buttons when at least one star is selected
        self.draw_psf_button.set_active(True)
        self.done_button.set_active(True)

    def compute_psf_contour(self, image, x, y, size=20):
        """Compute PSF contour for a star in an image."""
        half_size = size // 2
        region = image[
            max(0, y-half_size):min(image.shape[0], y+half_size),
            max(0, x-half_size):min(image.shape[1], x+half_size)
        ]
        
        mean, median, std = sigma_clipped_stats(region, sigma=3.0)
        threshold = median + 5.0 * std
        
        # Create contour and get vertices
        fig, ax = plt.subplots()
        cs = ax.contour(region, levels=[threshold])
        
        # Get vertices using the new recommended approach
        path = cs.allsegs[0][0]  # Get vertices from first level, first contour
        
        # Adjust vertices to absolute coordinates
        path[:, 0] += max(0, x-half_size)
        path[:, 1] += max(0, y-half_size)
        
        plt.close(fig)
        return path

    def draw_all_psf(self, event):
        """Draw PSF contours for selected stars."""
        first_file = self.frame_info['File'].iloc[0]
        
        # Compute and store PSF contours for first image
        self.psf_contours[first_file] = {}
        for star_num, (x, y) in self.star_positions.items():
            try:
                vertices = self.compute_psf_contour(self.image_data, x, y)
                self.psf_contours[first_file][star_num] = vertices
            except Exception as e:
                print(f"Error computing PSF for star {star_num}: {e}")
        
        # Draw contours
        for star_num, vertices in self.psf_contours[first_file].items():
            path = Path(vertices)
            patch = PathPatch(path, facecolor='none', edgecolor='red', linewidth=1)
            self.ax.add_patch(patch)
        
        self.fig.canvas.draw_idle()

    def finish_selection(self, event):
        """Complete star selection and close the figure."""
        plt.close(self.fig)
        self.compute_all_photometry()

    def compute_all_photometry(self):
        """Compute PSF photometry for all images and display results."""
        print("\nComputing PSF photometry for all images...")
        
        # Compute photometry for all images
        for filename in tqdm(self.images.keys(), desc="Processing images"):
            image = self.images[filename]
            # Readnoise
            N_R = float(self.frame_info.loc[self.frame_info['File'] == filename, 'Read Noise'].values[0])*0.37
            # Dark Current per pixel
            expt = float(self.frame_info.loc[self.frame_info['File']==filename, 'Exptime'].values[0])
            N_dark_pp = float(self.frame_info.loc[self.frame_info['File'] == filename, 'Dark Current'].values[0]) *0.37 *expt
            # Flat Noise per pixel
            flat_noise = float(self.frame_info.loc[self.frame_info['File'] == filename, 'Flat Noise'].values[0])*0.37
            # bkg_pp = self.calc_background(image)
            self.photometry[filename] = {}
            self.noise[filename] = {}
            
            # Compute contours and photometry for each star
            for star_num, (x, y) in self.star_positions.items():
                try:
                    vertices = self.compute_psf_contour(image, x, y)
                    
                    # Create mask from contour
                    img_y, img_x = np.mgrid[:image.shape[0], :image.shape[1]]
                    points = np.column_stack((img_x.ravel(), img_y.ravel()))
                    mask = Path(vertices).contains_points(points).reshape(image.shape)
                    # print(mask.shape)
                    # Calculate flux
                    flux = np.sum(image[mask])*0.37 
                    # Detector gain alwyas 0.37
                    # signal_noise = 0.37 * flux
                    # bkg_noise = np.sum(mask) * (0.37 * (bkg_pp +N_dark_pp) + (N_R**2))
                    # total_noise = np.sqrt(signal_noise + bkg_noise)
                    # total_noise = np.sqrt(flux+np.sum(mask)*(N_dark_pp + N_R**2 + flat_noise)) # 0.37 is the detector gain
                    total_noise = np.sqrt(flux+np.sum(mask)*(N_dark_pp + N_R**2)) # 0.37 is the detector gain
                    # total_noise
                    
                    # noise = np.sqrt(0.37 * flux + np.sum(mask) * (1 + (np.sum(mask)) *(0.37 * (bkgd_pp + N_dark_pp) + (N_R**2)))
                    self.photometry[filename][star_num] = flux
                    self.noise[filename][star_num] = total_noise
                except Exception as e:
                    print(f"\nError processing star {star_num} in {filename}: {e}")
                    self.photometry[filename][star_num] = np.nan
                    self.noise[filename][star_num] = np.nan

        # Create and display results DataFrame
        data = []
        for filename in self.images.keys():
            row = {'File': filename}
            row['Date-Obs'] = self.frame_info.loc[self.frame_info['File'] == filename, 'Date-Obs'].values[0]
            for star_num in self.star_positions.keys():
                x, y = self.star_positions[star_num]
                flux = self.photometry[filename][star_num]
                noise = self.noise[filename][star_num]

                row[f'Star_{star_num}_x'] = x
                row[f'Star_{star_num}_y'] = y
                row[f'Star_{star_num}_flux'] = flux
                # row[f'Star_{star_num}_contour'] = self.psf_contours[filename].get(star_num, None)
                row[f'Star_{star_num}_noise'] = noise
            
            data.append(row)
        
        results_df = pd.DataFrame(data)
        print("\nPhotometry Results:")
        print(results_df)
        # print(path)
        results_df.to_csv(head_dir+'psf_photometry_results.csv', index=False)
        return results_df



def get_frame_info(directories):
    """
    Extracts information from FITS file headers for reduced frames.
    """
    directories_list, file_list = [], []
    objects, dates, filters, exposure_times,dark_currents,read_noise,flat_noise = [], [], [], [], [], [],[]
    
    # Iterate through all directories
    for directory in directories:
        head_dir  = os.path.dirname(os.path.abspath(directory)) + '/' #Want to get dark current from uncertainties.csv in parent directory 
        
        # Get all FITS files in the directory
        fits_files = [f for f in os.listdir(directory) if f.endswith('.fits')]
        uncert = pd.read_csv(head_dir + 'Uncertainties.csv', header=None,sep='\s+')
        for file in fits_files:
            try:
                # Read the FITS header
                header = fits.getheader(os.path.join(directory, file))

                # Extract relevant header information
                directories_list.append(directory)
                file_list.append(file)
                objects.append(header.get('OBJECT', 'Unknown'))
                dates.append(header.get('DATE-OBS', 'Unknown'))
                filt = header.get('FILTER', 'Unknown')
                filters.append(filt)
                exp = header.get('EXPTIME', 'Unknown')
                exposure_times.append(exp)
                dark_currents.append(uncert.loc[uncert[0] == f'Dark_Current_{exp}s', 1].values[0])
                read_noise.append(uncert.loc[uncert[0] == 'Read_Noise', 1].values[0])
                flat_noise.append(uncert.loc[uncert[0] == f'Flat_{filt}_Noise', 1].values[0])

            except Exception as e:
                print(f"Error processing file {file} in {directory}: {e}")
                continue
        
        
    
        
    # Create a DataFrame with the extracted information
    reduced_frame_info = pd.DataFrame({'Directory': directories_list,
                                       'File': file_list,
                                       'Object': objects,
                                       'Date-Obs': dates,
                                       'Filter': filters,
                                       'Exptime': exposure_times,
                                       'Dark Current': dark_currents,
                                       'Read Noise': read_noise,
                                       'Flat Noise': flat_noise
                                       })

    return reduced_frame_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PSF Photometry Tool")
    parser.add_argument('-data', '--data', type=str, nargs='+', required=True,
                       help="Directories containing reduced images")
    args = parser.parse_args()
    path = args.data[0]
    head_dir  = os.path.dirname(os.path.abspath(path)) + '/' 
    frame_info_df = get_frame_info(args.data)
    # print(frame_info_df)
    frame_info_df.to_csv(head_dir+'frame_info.csv', index=False)
    psf_photometry = PSFPhotometry(frame_info_df)
    plt.show()