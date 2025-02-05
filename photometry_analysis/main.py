import numpy as np
import pandas as pd
import os
import sys
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval, ImageNormalize
from photutils.detection import DAOStarFinder, find_peaks
from photutils import aperture_photometry, CircularAperture, CircularAnnulus, datasets
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, Slider, ColumnDataSource, LabelSet, TextInput, CheckboxGroup, DataTable, TableColumn, Tabs, TabPanel
from bokeh.layouts import column, row
from bokeh.io import curdoc
from photometric_calculations import *



# Access the data directory from command-line arguments
data_dir = sys.argv[1]

# Read in the first image from the parsed directory
file_list = os.listdir(data_dir)

image_data_path = os.path.join(data_dir, file_list[0])

# Define the function to display the input images
def image_display_tab(data_dir):
    """
    Creates the tab for displaying images with a slider and a selectable DataTable.
    """

    # Get the list of FITS files in the directory
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.fits')]

    if not file_list:
        return None  # Return None if no FITS files are found

    # Load the first image
    image_data_path = os.path.join(data_dir, file_list[0])
    image_data = fits.getdata(image_data_path)

    # ZScale normalize the image data
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(image_data)
    stretched_image = np.clip(image_data, vmin, vmax)

    # Create the figure (add title argument and alignment)
    p = figure(width=image_data.shape[0],
              tools="pan,wheel_zoom,box_zoom,reset",
              x_axis_location=None,
              y_axis_location=None,
              title=file_list[0],
              title_location="above")
    p.image(image=[stretched_image],
             x=0,
             y=0,
             dw=image_data.shape[1],
             dh=image_data.shape[0])
    p.title.align = "center"

    # Remove grid lines
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    # Create a slider
    slider = Slider(start=0,
                    end=len(file_list) - 1,
                    value=0,
                    step=1,
                    title="Image Index")

    # Get the DataFrame with FITS information
    reduced_frame_info = get_frame_info([data_dir])

    # Create a ColumnDataSource from the DataFrame
    source = ColumnDataSource(reduced_frame_info)

    # Define the columns for the DataTable
    columns = [
        TableColumn(field="File", title="File"),
        TableColumn(field="Object", title="Object"),
        TableColumn(field="Date-Obs", title="Date-Obs"),
        TableColumn(field="Filter", title="Filter"),
        TableColumn(field="Exptime", title="Exptime"),
    ]

    # Create the DataTable (add selectable argument)
    data_table = DataTable(source=source,
                          columns=columns,
                          width=600,
                          height=600,
                          selectable="checkbox")

    # Define the callback function for the slider
    def update_image(attrname, old, new):
        # Get the active files from the data_table
        active_indices = data_table.selected.indices
        active_files = [file_list[i] for i in active_indices]

        # Update the file_list based on the active files
        file_list[:] = active_files

        # Update the slider's end value
        slider.end = len(file_list) - 1

        # If the current slider value is out of range, reset it to 0
        if slider.value >= len(file_list):
            slider.value = 0

        # Load the new image
        if file_list:  # Check if there are any active files
            image_data_path = os.path.join(data_dir, file_list[slider.value])
            image_data = fits.getdata(image_data_path)

            # ZScale normalize the image data
            zscale = ZScaleInterval()
            vmin, vmax = zscale.get_limits(image_data)
            stretched_image = np.clip(image_data, vmin, vmax)

            # Update the image and title
            p.image(image=[stretched_image],
                    x=0,
                    y=0,
                    dw=image_data.shape[1],
                    dh=image_data.shape[0])
            p.title.text = file_list[slider.value]

    # Add the callback to the slider
    slider.on_change('value', update_image)

    # Create the layout for the image display
    image_display_layout = row(
        column(slider, p),  # Image display on the left
        column(data_table)#, width=600)  # Table on the right
    )

    # Create the TabPanel
    tab = TabPanel(child=image_display_layout, title="Image Display")

    return tab, data_table




def display_bokeh_image(image_data_path):

    # Extract the image stats
    image_data, mean, median, stddev = image_stats_out(image_data_path)

    # ZScale normalize the image data
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(image_data)
    stretched_image = np.clip(image_data, vmin, vmax)

    # Find initial peaks
    peaks_df = find_star_peaks(image_data, threshold=3, median=median, std=stddev)

    # Select the top 10 brightest peaks
    peaks_df = peaks_df.nlargest(10, 'peak_value')

    source = ColumnDataSource(peaks_df)

    # Define the bokeh object to display the image
    plot = figure(width=image_data.shape[0], tools="pan,wheel_zoom,box_zoom,reset,hover", x_axis_location=None, y_axis_location=None)
    plot.image(image=[stretched_image], x=0, y=0, dw=image_data.shape[1], dh=image_data.shape[0])

    # Add red crosses for peak positions (using a separate source for the crosses)
    crosses_source = ColumnDataSource(dict(x=peaks_df['x_peak'].tolist(),
                                           y=peaks_df['y_peak'].tolist()))
    plot.cross(x='x',
               y='y',
               source=crosses_source,
               size=10,
               color='lime',
               line_width=2)
    
    # Add labels for star IDs
    labels = LabelSet(x='x_peak',
                      y='y_peak',
                      text='id',
                      source=source,
                      x_offset=-20,
                      y_offset=20,
                      text_color='lime')
    plot.add_layout(labels)

    # Remove grid lines
    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = None

    # Define the HoverTool instance
    hover = plot.select(dict(type=HoverTool))

    # Format the tooltip to display x, y, and pixel value
    hover.tooltips = [("x", "$x"),
                      ("y", "$y"),
                      ("value", "@image{0.000}")]

    # Create a slider for the threshold
    threshold_slider = Slider(start=1, end=5, value=3, step=1, title="Peak Finding Threshold (stddev)")

    # Create a TextInput to sellect the number of peaks
    text_input = TextInput(value="10", title="Number of brightest peaks:")

    # Add the callback function
    def update_data(attrname, old, new):

        # Get the current slider values
        threshold = threshold_slider.value

        # Find the new star peaks
        peaks_df = find_star_peaks(image_data, threshold=threshold, median=median, std=stddev)

        # Get the number of peaks from the TextInput widget
        try:
            n_peaks = int(text_input.value)

        except ValueError:
            # Default to 10 if the input is invalid
            n_peaks = 10 

        # Select the top n_peaks brightest peaks
        peaks_df = peaks_df.nlargest(n_peaks, 'peak_value')

        # Update the data of the existing ColumnDataSource
        source.data = peaks_df.to_dict('list')

        # Update the crosses source
        crosses_source.data = dict(x=peaks_df['x_peak'].tolist(), y=peaks_df['y_peak'].tolist())

        # Remove the old LabelSet
        if plot.renderers:  # Check if there are any renderers
            for renderer in plot.renderers:
                if isinstance(renderer, LabelSet):
                    plot.renderers.remove(renderer)

        # Add the updated LabelSet
        labels = LabelSet(x='x_peak',
                          y='y_peak',
                          text='id',
                          source=source,
                          x_offset=-25,
                          y_offset=25,
                          text_color='lime')
        plot.add_layout(labels)

    # Add the callback to the widgets
    threshold_slider.on_change('value', update_data)
    text_input.on_change('value', update_data)

    # Create a layout
    layout = column(row(threshold_slider, text_input), plot)

    return layout


# Create the tabs
tab1, selected_data = image_display_tab(data_dir)
tab2 = TabPanel(child=display_bokeh_image(image_data_path), title="Peak Finding")


# Combine tabs into Tabs layout
tabs = Tabs(tabs=[tab1, tab2])

# Add the tabs to the document
curdoc().add_root(tabs)