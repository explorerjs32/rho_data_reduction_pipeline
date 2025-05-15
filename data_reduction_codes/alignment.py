import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from astropy.visualization import ImageNormalize, ZScaleInterval


class ImageAlignmentTool:
    def __init__(self, template_image, target_image, template_name="", target_name=""):
        """
        Initialize the alignment tool with template and target images.
        
        Parameters:
        -----------
        template_image : numpy.ndarray
            Reference image for alignment
        target_image : numpy.ndarray
            Image to be aligned with template
        template_name : str, optional
            Name of the template image for display
        target_name : str, optional
            Name of the target image for display
        """
        # Store the images and names
        self.template = template_image
        self.target = target_image
        self.template_filename = template_name
        self.target_filename = target_name
        
        # Initialize point lists
        self.template_points = []
        self.target_points = []
        self.selecting_template = True

        # Create the figure and subplots in a 2x2 grid
        self.fig = plt.figure(figsize=(20, 15))
        gs = self.fig.add_gridspec(2, 2, hspace=0.2, wspace=0.2)

        # Create subplots for the template, target, and overlay images
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[1, 0])
        self.button_panel = self.fig.add_subplot(gs[1, 1])

        # Remove the axes labels and ticks for the image panels
        self.ax1.axis('off')
        self.ax2.axis('off')
        self.ax3.axis('off')

        # Remove the axis for the button panel
        self.button_panel.axis('off')

        # Connect the scroll event to the zoom function
        self.scroll_cid = self.fig.canvas.mpl_connect('scroll_event', self.zoom_image)

        # Add reference to navigation toolbar
        self.toolbar = plt.get_current_fig_manager().toolbar

        self.setup_plot()

    def setup_plot(self):
        """
        Set up the initial layout and button display.
        """
        # Normalize the images to ZScale
        target_norm = ImageNormalize(self.target, interval=ZScaleInterval())
        template_norm = ImageNormalize(self.template, interval=ZScaleInterval())

        # Display the images
        self.ax1.imshow(self.template, origin='lower', cmap='gray', norm=template_norm)
        self.ax2.imshow(self.target, origin='lower', cmap='gray', norm=target_norm)
        
        # Display overlay
        self.ax3.imshow(self.template, origin='lower', cmap='Blues', alpha=0.5, norm=template_norm)
        self.ax3.imshow(self.target, origin='lower', cmap='Reds', alpha=0.5, norm=target_norm)
        
        # Create proxy artists for legend
        template_patch = plt.Rectangle((0, 0), 1, 1, fc='blue', alpha=0.5)
        target_patch = plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.5)
        
        # Add legend with proxy artists
        self.ax3.legend([template_patch, target_patch], 
                        ['Template', 'Target'],
                        ncols=2,
                        bbox_to_anchor=(0.8, 0.0))

        # Set titles
        self.ax1.set_title(f"{self.template_filename} (Template Image)", size=10)
        self.ax2.set_title(f"{self.target_filename} (Target Image)", size=10)
        self.ax3.set_title("Overlay of Template and Target")

        # Add buttons for interaction
        self.add_buttons()

        # Connect the click event to the function
        self.click_cid = self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def zoom_image(self, event):
        """
        Zoom in and out of the images using the scroll wheel.
        """
        if event.inaxes in [self.ax1, self.ax2, self.ax3]:
            current_ax = event.inaxes
            x, y = event.xdata, event.ydata
            scale_factor = 1.25 if event.button == 'down' else 0.75
            
            # Get current x and y limits
            cur_xlim = current_ax.get_xlim()
            cur_ylim = current_ax.get_ylim()
            
            # Calculate the new limits
            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            
            # Calculate center points
            xcenter = (cur_xlim[0] + cur_xlim[1]) / 2
            ycenter = (cur_ylim[0] + cur_ylim[1]) / 2
            
            # Set new limits
            current_ax.set_xlim([xcenter - new_width / 2, xcenter + new_width / 2])
            current_ax.set_ylim([ycenter - new_height / 2, ycenter + new_height / 2])
            
            # Get the original image dimensions based on which image is being zoomed
            if current_ax == self.ax1:
                image_data = self.template
            elif current_ax == self.ax2:
                image_data = self.target
            else:  # For overlay plot (ax3), use template dimensions
                image_data = self.template
                
            image_height, image_width = image_data.shape
            
            # Set the limits to the original image dimensions if zoomed out too much
            if new_width > image_width:
                current_ax.set_xlim(0, image_width)
            if new_height > image_height:
                current_ax.set_ylim(0, image_height)
                
            self.fig.canvas.draw_idle()

    def add_buttons(self):
        """
        Add interactive buttons to the plot.
        """
        # Create button axes in the bottom right panel
        self.align_button_ax = plt.axes([0.65, 0.35, 0.25, 0.05])
        self.reset_button_ax = plt.axes([0.65, 0.25, 0.25, 0.05])
        self.done_button_ax = plt.axes([0.65, 0.15, 0.25, 0.05])

        # Create the buttons
        self.align_button = Button(self.align_button_ax, "Align")
        self.reset_button = Button(self.reset_button_ax, "Reset")
        self.done_button = Button(self.done_button_ax, "Done")

        # Add button callbacks
        self.align_button.on_clicked(self.align_images)
        self.reset_button.on_clicked(self.reset)
        self.done_button.on_clicked(self.close_figure)

    def on_click(self, event):
        """
        Handle click events for point selection.
        """
        # Check if any navigation toolbar tool is active
        if self.toolbar.mode != '':
            return
            
        if event.inaxes == self.ax1 and self.selecting_template:
            # Get the coordinates of the clicked point in the template image
            x, y = int(event.xdata), int(event.ydata)
            self.template_points.append((x, y))
            self.ax1.plot(x, y, 'r+', markersize=5)
        
        elif event.inaxes == self.ax2:
            # Get the coordinates of the clicked point in the target image
            x, y = int(event.xdata), int(event.ydata)
            self.target_points.append((x, y))
            self.ax2.plot(x, y, 'r+', markersize=5)

        plt.draw()

    def calculate_shift(self):
        """
        Caluclate the average shift between the template and target points.
        """
        dx = []
        dy = []
        for tp, tgp in zip(self.template_points, self.target_points):
            dx.append(tgp[0] - tp[0])
            dy.append(tgp[1] - tp[1])

        return int(np.mean(dx)), int(np.mean(dy))

    def align_images(self, event):
        """
        Align images using the calculated shift.
        """
        # Calculate the average shift between the template and target points
        shift = self.calculate_shift()

        # Roll the target image to align it with the template
        rolled_target = np.roll(np.roll(self.target, -shift[1], axis=0), -shift[0], axis=1)

        # Normalize the rolled target image to ZScale
        rolled_target_norm = ImageNormalize(rolled_target, interval=ZScaleInterval())

        # Update the overlay plot with the aligned image
        self.ax3.clear()
        self.ax3.imshow(self.template, origin='lower', cmap='Blues', alpha=0.5, norm=ImageNormalize(self.template, interval=ZScaleInterval()))
        self.ax3.imshow(rolled_target, origin='lower', cmap='Reds', alpha=0.5, norm=rolled_target_norm)

        # Create proxy artists for legend
        template_patch = plt.Rectangle((0, 0), 1, 1, fc='blue', alpha=0.5)
        target_patch = plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.5)
        
        # Add legend with proxy artists
        self.ax3.legend([template_patch, target_patch], 
                        ['Template', 'Target'],
                        ncols=2,
                        bbox_to_anchor=(0.8, 0.0))
        
        # Remove the axes labels and ticks for the image panels
        self.ax1.axis('off')
        self.ax2.axis('off')
        self.ax3.axis('off')
        
        self.ax3.set_title("Aligned Overlay")
        plt.draw()

    def reset(self, event):
        """
        Reset all selections and plots.
        """
        self.template_points = []
        self.target_points = []
        self.selecting_template = True
        self.align_button.color = '0.85'

        # CLear plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # Redraw original images with overlay
        self.ax1.imshow(self.template, origin='lower', cmap='gray', 
                        norm=ImageNormalize(self.template, interval=ZScaleInterval()))
        self.ax2.imshow(self.target, origin='lower', cmap='gray', 
                        norm=ImageNormalize(self.target, interval=ZScaleInterval()))
        self.ax3.imshow(self.template, origin='lower', cmap='Blues', 
                        alpha=0.5, norm=ImageNormalize(self.template, interval=ZScaleInterval()))
        self.ax3.imshow(self.target, origin='lower', cmap='Reds', 
                        alpha=0.5, norm=ImageNormalize(self.target, interval=ZScaleInterval()))

        # Create proxy artists for legend
        template_patch = plt.Rectangle((0, 0), 1, 1, fc='blue', alpha=0.5)
        target_patch = plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.5)
        
        # Add legend with proxy artists
        self.ax3.legend([template_patch, target_patch], 
                        ['Template', 'Target'],
                        ncols=2,
                        bbox_to_anchor=(0.8, 0.0))
        
        # Remove the axes labels and ticks for the image panels
        self.ax1.axis('off')
        self.ax2.axis('off')
        self.ax3.axis('off')

        # Reset titles
        self.ax1.set_title(f"{self.template_filename} (Template Image)")
        self.ax2.set_title(f"{self.target_filename} (Target Image)")
        self.ax3.set_title("Overlay of Template and Target")

        plt.draw()

    def get_aligned_image(self):
        """
        Returns the aligned image after manual alignment.
        
        Returns:
        --------
        numpy.ndarray or None
            The aligned image if alignment was successful, None otherwise
        """
        shift = self.calculate_shift()
        if shift:
            return np.roll(np.roll(self.target, -shift[1], axis=0), -shift[0], axis=1)
        return None

    def close_figure(self, event):
        """
        Close the figure when Done button is clicked.
        """
        plt.close(self.fig)
