import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from astropy.io import fits
from astropy.visualization import ImageNormalize, ZScaleInterval, LogStretch, LinearStretch
import argparse
import os


class ImageAlignmentTool:
    def __init__(self, template_image, target_image):
        """
        Initialize the alignment tool with template and target images.
        
        Parameters:
        -----------
        template_image : numpy.ndarray
            Reference image for alignment
        target_image : numpy.ndarray
            Image to be aligned with template
        """
        self.template = template_image
        self.target = target_image
        self.template_points = []
        self.target_points = []
        self.current_points = []
        self.selecting_template = True

        # Create the figure and subplots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
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
        self.ax3.imshow(self.target, origin='lower', cmap='Reds', alpha=0.5, norm=target_norm)
        self.ax3.imshow(self.template, origin='lower', cmap='Blues', alpha=0.5, norm=template_norm)

        # Set titles
        self.ax1.set_title("Template Image")
        self.ax2.set_title("Target Image")
        self.ax3.set_title("Overlay of Template and Target")

        # Add buttons for interaction
        self.add_buttons()

        # Connect the click event to the function
        self.click_cid = self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def add_buttons(self):
        """
        Add interactive buttons to the  plot for user interaction.
        """
        # Create button axes
        self.button_ax = plt.axes([0.8, 0.05, 0.1, 0.04])
        self.reset_ax = plt.axes([0.6, 0.05, 0.1, 0.04])

        # Create the buttons
        self.align_button = Button(self.button_ax, "Align")
        self.reset_button = Button(self.reset_ax, "Reset")

        # Add button callbacks
        self.align_button.on_clicked(self.align_images)
        self.reset_button.on_clicked(self.reset)

    def on_click(self, event):
        """
        Handle click events for point selection.
        """
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

        # Redraw original images
        self.ax1.imshow(self.template, origin='lower', cmap='gray', norm=ImageNormalize(self.template, interval=ZScaleInterval()))
        self.ax2.imshow(self.target, origin='lower', cmap='gray', norm=ImageNormalize(self.target, interval=ZScaleInterval()))
        self.ax3.imshow(self.target, origin='lower', cmap='Reds', alpha=0.5, norm=ImageNormalize(self.target, interval=ZScaleInterval()))
        self.ax3.imshow(self.template, origin='lower', cmap='Blues', alpha=0.5, norm=ImageNormalize(self.template, interval=ZScaleInterval()))

        # Reset titles
        self.ax1.set_title("Template Image")
        self.ax2.set_title("Target Image")
        self.ax3.set_title("Overlay of Template and Target")

        plt.draw()


def main():
    """
    Main functio n to run the image alignment process.
    """
    parser = argparse.ArgumentParser(description="Image Alignment Tool")
    parser.add_argument('-i', '--input_dir', type=str, required=True, help="Input FITS file directory")

    args = parser.parse_args()

    # Load the images
    image_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.fits')])

    # Define the template image (first image in the list)
    template_image = fits.getdata(os.path.join(args.input_dir, image_files[0]))

    # Run the alignment tool
    allignment_tool = ImageAlignmentTool(template_image, fits.getdata(os.path.join(args.input_dir, image_files[6])))
    plt.show()

if __name__ == "__main__":
    main()
