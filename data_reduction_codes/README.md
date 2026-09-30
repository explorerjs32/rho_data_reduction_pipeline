# Image Reduction
This tool is designed to automatically perform the reduction and calibration of raw images, align them to a reference frame, save the reduced images, and frame information taken from the `HEADER` of the `.fits` files.

## Usage
The data reduction process is divided in two different steps: (1) sorting raw observations, and (2) reducing raw images for one or more objects. 

### Sorting Raw Images
Once you have downloaded the raw images from RHO, you can parse in the data directory to into the `sort_observations.py` script. This script takes in a single argument called `--dir` or `-d` for short, which should point to the directory that has the raw images. 

You can run the script as follows:

`python sort_observations.py -d <path_to_raw_data>`


**Note:** You should parse in a directory rather than a list of files or a single file for the code to work properly.

The output of this directory will be a copy of the raw image files inside the parsed directory, but they will be re-organized into four different sub-directories based on the frame type of the image (Light, Dark, Flat, and Bias). The light frames sub-directory will also stored the re-organized frames by the object name, and there will be an additional sub-directory for each object individually.

These sub-directories will be usefull for the next step, which is the image reduction process.

There is an optional argument `--del_OG` that can be added after specified directory path, which will delete the original files after sorting them into subdirectories. This option is not typically recommended in case of sorting mistakes, but users low on disk space may not want to keep multiple copies of large datasets.

**Coordinate Search:** This script also checks if there is are keywords for telescope or target `RA` and `DEC` in the light frames fits headers when sorting by object, which are required for photometry. By default these should be included in the image fits header from RHO, but for times these are unavailable, this function will allow users to add these back to the image headers using `astroquery` . If these are missing or blank from the fits image header, the code will first query the SIMBAD database by the object name given in the header of the image. If the query is successful, the user will be shown the coordinates it returns, and prompted to enter `yes` or `no` in the command line to confirm these are correct for the given object. If `yes` is entered, the code will update the fits headers for all sorted copied frames corresponding to this object. 

If the user enters `no`, or the query is unsuccessful with the object name provided in the header, the user will be prompted with three options to tell the code how to proceed in filling in the `RA` and `DEC` keywords in the header. These options are as follows and can be selected by entering a number `1/2/3` into the command line, corrsponding to the desired option: 


`1. Search with a different target name`: User will manually enter the target name in the command line corresponding to the object. Useful if object names in headers aren't directly queriable in simbad (ex. object name in header for the exoplanet target is "GJ860B" will have no results from a SIMBAD query, but user can enter the star name "GJ 860" to successfully retrieve the coordinates for this target.)

`2. Manually input coordinates`: User will manually enter the target `RA` and `DEC` in sexagesimal format, and shown examples of the desired format (` RA:  12 34 56.78 or 12:34:56.78 or 12h34m56.78s`, `DEC: +12 34 56.7 or +12:34:56.7 or +12d34m56.7s`). This option will be useful if SIMBAD queries are unsuccessful for a given target, the astroquery server is down/overloaded, or if the user already has the coordinates on hand.

`3. Skip (leave coordinates empty)`: This will add fields for `RA` and `DEC` in the fits header, but leave them blank. This option is useful for users that are testing the pipeline or just sorting observations, for test frames where object name is unkown, or for users not looking to perform photometry. Note that if `RA` and `DEC` are left blank at this step, the user will be prompted with these options again when running `image_reduction.py` or `image_reduction_interact_select.py` below. This can also be a useful option then for anyone wanting to double check their object coordinates before entering them at the reduction phase. 

The image headers will only be updated for the copied and sorted frames, not the original raw frames, so that if the user needs correct the object coordinates or makes a mistake in entering the object coordinates, they can simply run `sort_observations.py` on the raw files again to go through these steps again and assign the correct coordinates without manually modifying each frame themselves. 

### Reducing Raw Images
After the raw images have been classified into their different sub-directories. You can run the script `image_reduction.py` by parsing in these directories. This script can also be found in the `./data_reduction/` directory of this repository.

Similarly to the `sort_observations.py` script, you will be required to parse in different in order to get the expected outcome. These arguments are:

* `-l`: Directory containing the raw light frames from an object (e.g. `<raw_data_dir>/Light/<obj_name>`)
* `-d`: Directory containing the dark frames (e.g. `<raw_data_dir>/Dark/`)
* `-f`: Dierctory containing the flat frames (e.g. `<raw_data_dir>/Flat/`)
* `-b`: Directory containing the bias frames (e.g. `<raw_data_dir>/Bias/`)
* `-B`: *(optional boolean)* `True` or `False` argument to perform sky background subtraction. Default value is set to `True`
* `-O`: *(optional boolean)* `True` or `False` argument to allow fits file overwritting. Default value is set to `False`
* `-o`: Directory where you want the the reduced files to be stored as well as the auxiliary files created by the data reduction script.

So, when running this code it should look like the following:

`python image_reduction.py -l <raw_data_dir>/Light/<obj_name> -d <raw_data_dir>/Dark/ -f <raw_data_dir>/Flat/ -b <raw_data_dir>/Bias/ -o <output_dir>`

By running this code using the above example, the data reduction pipeline will perform sky background subtraction since that is the default setting. To skip this step, you can run the code as follows:

`python image_reduction.py -l <raw_data_dir>/Light/<obj_name> -d <raw_data_dir>/Dark/ -f <raw_data_dir>/Flat/ -b <raw_data_dir>/Bias/ -B False -o <output_dir>`

The output of this script will be the following:

* A directory named `reduced` containing: 
    * A sub-directory with the object name that was reduced. This sub-directory will have the reduced light frames for that respective object.
    * A file named `data_reduction_report.txt` containing a detailed process of the data reduction steps for each frame individually (i.e. what settings were used to collect the individual raw light frames, and what callibration frames were used to reduce them).
    * A file named `Uncertainties.csv` containing different instrumental noise uncertainties from the used callibration frames, which will be later used during the photometric calculations.

You can also run this script by parsing in light frames for more than one object, or raw files from different nights. For instance, if you observed two objects during night one, and you collected callibration frames across two different nights, then you can run the script as follows:

`python image_reduction.py -l <raw_data_dir_1>/Light/<obj_name_1> <raw_data_dir_1>/Light/<obj_name_2> -d <raw_data_dir_1>/Dark/ <raw_data_dir_2>/Dark/ -f <raw_data_dir_1>/Flat/ <raw_data_dir_2>/Flat/ -b <raw_data_dir_1>/Bias/ -o <output_dir>`

**Note 1:** If you want to reduce individual objects separatelly by running the script several times, it is recommended to select different output directories for each run, as the output files will be over-writen each time. 

**Note 2:** If you want to perform background subtraction for one object and not for the other(s), it is recommended that you run this pipeline separately for each object.

**Note 3:** If `RA` and `DEC` keywords of the target frame are still missing or left blank despite the earlier check, the user will be prompted to fill these once again as described in the *Sorting Raw Images* section above. 

### Interactive Image Reduction
After the raw images have been classified into their different sub-directories. You can run the script `image_reduction_interact_select.py` . This script can also be found in the `./data_reduction/` directory of this repository.

This script functionally operates the same as `image_reduction.py` described above, but rather than manually specifying the directory paths, a GUI window will appear, allowing you to interactively select the paths within your finder or file explorer to the calibration and object frames. There is currently an option to add additional light frame directories for reducing multiple objects with the same calibration data, as well as options to add additional dark or flat frame directories if you need to use calibration frames taken separately from your main observations. 

The notes related to the existing raw image reduction functions apply to this script as well. 

--- 

## Interactive Manual Alignment Tool

The Interactive Manual Alignment Tool provides a graphical interface for aligning astronomical images when automatic alignment fails. This tool is essential for ensuring high-quality data reduction, especially in cases where images have large offsets, low signal-to-noise, or artifacts that prevent automated routines from working reliably.

### When Is This Tool Triggered?

The tool is automatically invoked by the pipeline when the standard image alignment (typically using `astroalign` or similar algorithms) cannot successfully register a target image to the template. This may occur due to:

- Insufficient or ambiguous features for matching (e.g., few stars, cosmic rays, or artifacts).
- Large shifts or rotations between images.
- Unusual image distortions or defects.

When such a scenario is detected, the pipeline launches the Interactive Manual Alignment Tool, pausing automated processing and allowing the user to intervene.

### User Interface Overview

Upon activation, the tool displays a window with the following layout:

- **Top Left:** Template (reference) image.
- **Top Right:** Target (image to be aligned).
- **Bottom Left:** Overlay of template and target images for visual comparison.
- **Bottom Right:** Control panel with interactive buttons.

Each image panel supports zooming (mouse scroll wheel) and panning (matplotlib toolbar) for precise navigation.

### How to Use the Tool

#### 1. Zooming and Panning

- **Zoom:**  
  Use the mouse scroll wheel over any image panel to zoom in or out for more precise point selection.
- **Pan:**  
  Use the navigation toolbar at the bottom of the window to pan across the images.

#### 2. Selecting Correspondence Points

- **Template Points:**  
  Click on the Template image (top left) to select reference points. Each click marks a red "+" at the selected location.
- **Target Points:**  
  Click on the Target image (top right) to select the corresponding points in the image you wish to align. Each click marks a red "+".
- **Best Practice:**  
  Select the same number of points in both images, ensuring each pair corresponds to the same astronomical feature (e.g., a star or bright object). At least two pairs are recommended for robust alignment.

#### 3. Overlay Visualization

- The Overlay panel (bottom left) shows both images superimposed (template in blue, target in red), allowing you to visually assess the alignment.

### Button Functions

- **Align:**  
  Computes the average shift between the selected template and target points, applies this shift to the target image, and updates the overlay for preview.  
  *Use this after selecting corresponding points in both images.*

- **Reset:**  
  Clears all selected points and resets the images and overlay to their original state.  
  *Use this if you want to start the point selection process over.*

- **Accept As Is:**  
  Accepts the target image without any alignment. The pipeline will proceed using the original, unaligned image.  
  *Use this if you believe the image does not require alignment or if alignment is not possible.*

- **Ignore Image:**  
  Skips the current image entirely. The pipeline will not use this image in further processing.  
  *Use this if the image is unusable or too problematic to align.*

- **Done:**  
  Closes the alignment tool window and returns control to the pipeline.  
  *Use this after you have finished aligning or making your selection.*

### Workflow Summary

1. The tool opens automatically if automatic alignment fails.
2. Select corresponding points in both the template and target images.
3. Click **Align** to preview the alignment in the overlay.
4. If satisfied, click **Done** to save the aligned image.
5. If not, use **Reset** to try again, **Accept As Is** to keep the original, or **Ignore Image** to skip.
6. The pipeline continues with the next image or step.

### Tips for Effective Alignment

- Select at least two well-separated, easily identifiable features for best results.
- Use zoom and pan to improve accuracy when clicking on features.
- If the overlay looks misaligned after pressing **Align**, try resetting and selecting more accurate or additional points.

### Troubleshooting

- If the tool does not appear, ensure your environment supports PyQt5 and matplotlib interactive backends.
- If you accidentally close the window, rerun the pipeline step to trigger the tool again.

This tool ensures robust and user-friendly manual alignment, allowing you to process even the most challenging astronomical images with confidence