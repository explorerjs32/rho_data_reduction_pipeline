import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import simple_norm, ZScaleInterval, AsinhStretch, SqrtStretch, ImageNormalize,MinMaxInterval
from matplotlib.widgets import RangeSlider
from matplotlib.gridspec import GridSpec
import argparse 


"""Display widget to visualize an RGB composite image
generated from aligned R,G,B master FITS images with interactive sliders for color scaling.

Current version is a standalone script that can be run from the command line.

Usage: python RGBwidget.py r_aligned.fits g_aligned.fits b_aligned.fits

The script will display the RGB composite image and histograms of the R, G, B channels.

Three sliders are provided to adjust the color scaling for each channel.

ZScale normalization is used to set the initial scaling range for each channel.

The script requires the astropy and matplotlib libraries to be installed.

"""

# Load FITS files
def load_fits(file):
    
    return fits.getdata(file).astype(float)


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Load and visualize RGB FITS images.")
    parser.add_argument("r_image", type=str, help="Path to the Red channel FITS image.")
    parser.add_argument("g_image", type=str, help="Path to the Green channel FITS image.")
    parser.add_argument("b_image", type=str, help="Path to the Blue channel FITS image.")
    return parser.parse_args()

# Parse the arguments
args = parse_args()

# Load the specified FITS files
R_img = load_fits(args.r_image)
G_img = load_fits(args.g_image)
B_img = load_fits(args.b_image)

# Compute percentiles for better scaling
r_min, r_max = np.percentile(R_img, [1, 99])
g_min, g_max = np.percentile(G_img, [1, 99])
b_min, b_max = np.percentile(B_img, [1, 99])

def ZScale(data):
    # interval = ZScaleInterval()
    # return interval(data)
    return ImageNormalize(data, interval=ZScaleInterval())

r_min, r_max = ZScale(R_img).vmin, ZScale(R_img).vmax
g_min, g_max = ZScale(G_img).vmin, ZScale(G_img).vmax
b_min, b_max = ZScale(B_img).vmin, ZScale(B_img).vmax

# Function to normalize images
def normalize(img, vmin, vmax):
    return np.clip((img - vmin) / (vmax - vmin), 0, 1)

# Initial normalized images
R_scaled = normalize(R_img, r_min, r_max)
G_scaled = normalize(G_img, g_min, g_max)
B_scaled = normalize(B_img, b_min, b_max)

# Create the combined RGB image
RGB_img = np.dstack([R_scaled, G_scaled, B_scaled])

# Set up figure with GridSpec for better layout
fig = plt.figure(figsize=(16, 7))  # Increased width for better spacing
gs = GridSpec(3, 5, width_ratios=[6, 4, 4, 4, 4], height_ratios=[1, 1, 1])  

# Histograms (Left Column, Wide & Stacked)
ax_r_hist = fig.add_subplot(gs[0, 0])
ax_g_hist = fig.add_subplot(gs[1, 0])
ax_b_hist = fig.add_subplot(gs[2, 0])

# Composite RGB Image (Right, Large)
ax_rgb = fig.add_subplot(gs[:, 1:])  # Takes up all rows in the right four columns

# Display the RGB image
RGB_im = ax_rgb.imshow(RGB_img, origin='lower')
ax_rgb.axis("off")
ax_rgb.set_title("RGB Composite")

# Red Histogram (Wider)
ax_r_hist.hist(R_img.flatten(), bins=100, color='red', range=(r_min, r_max))
ax_r_hist.set_xlim(r_min, r_max)
ax_r_hist.set_yticks([])
ax_r_hist.set_title("Red", fontsize=10)

# Green Histogram (Wider)
ax_g_hist.hist(G_img.flatten(), bins=100, color='green', range=(g_min, g_max))
ax_g_hist.set_xlim(g_min, g_max)
ax_g_hist.set_yticks([])
ax_g_hist.set_title("Green", fontsize=10)

# Blue Histogram (Wider)
ax_b_hist.hist(B_img.flatten(), bins=100, color='blue', range=(b_min, b_max))
ax_b_hist.set_xlim(b_min, b_max)
ax_b_hist.set_yticks([])
ax_b_hist.set_title("Blue", fontsize=10)

# Move sliders down to avoid overlap
slider_ax_r = fig.add_axes([0.15, 0.07, 0.7, 0.03])  # Lowered from 0.15 to 0.08
r_slider = RangeSlider(slider_ax_r, "Red", r_min, r_max, valinit=(r_min, r_max))
r_slider.poly.set_facecolor("red")

slider_ax_g = fig.add_axes([0.15, 0.04, 0.7, 0.03])  # Lowered
g_slider = RangeSlider(slider_ax_g, "Green", g_min, g_max, valinit=(g_min, g_max))
g_slider.poly.set_facecolor("green")

slider_ax_b = fig.add_axes([0.15, 0.01, 0.7, 0.03])  # Lowered
b_slider = RangeSlider(slider_ax_b, "Blue", b_min, b_max, valinit=(b_min, b_max))
b_slider.poly.set_facecolor("blue")

# Create vertical lines on the histograms
lower_limit_line_r = ax_r_hist.axvline(r_slider.val[0], color='k')
upper_limit_line_r = ax_r_hist.axvline(r_slider.val[1], color='k')

lower_limit_line_g = ax_g_hist.axvline(g_slider.val[0], color='k')
upper_limit_line_g = ax_g_hist.axvline(g_slider.val[1], color='k')

lower_limit_line_b = ax_b_hist.axvline(b_slider.val[0], color='k')
upper_limit_line_b = ax_b_hist.axvline(b_slider.val[1], color='k')

# Update functions for each slider
def update_R(val):
    global R_scaled, RGB_img
    R_scaled = normalize(R_img, val[0], val[1])
    RGB_img = np.dstack([R_scaled, G_scaled, B_scaled])
    RGB_im.set_data(RGB_img)
    lower_limit_line_r.set_xdata([val[0], val[0]])
    upper_limit_line_r.set_xdata([val[1], val[1]])
    fig.canvas.draw_idle()

def update_G(val):
    global G_scaled, RGB_img
    G_scaled = normalize(G_img, val[0], val[1])
    RGB_img = np.dstack([R_scaled, G_scaled, B_scaled])
    RGB_im.set_data(RGB_img)
    lower_limit_line_g.set_xdata([val[0], val[0]])
    upper_limit_line_g.set_xdata([val[1], val[1]])
    fig.canvas.draw_idle()

def update_B(val):
    global B_scaled, RGB_img
    B_scaled = normalize(B_img, val[0], val[1])
    RGB_img = np.dstack([R_scaled, G_scaled, B_scaled])
    RGB_im.set_data(RGB_img)
    lower_limit_line_b.set_xdata([val[0], val[0]])
    upper_limit_line_b.set_xdata([val[1], val[1]])
    fig.canvas.draw_idle()

r_slider.on_changed(update_R)
g_slider.on_changed(update_G)
b_slider.on_changed(update_B)

plt.show()