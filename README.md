# RETRHO Data Reduction Pipeline

Welcome to the RETRHO data reduction pipeline repository! Here, you will find the necesary tools to automatically callibrate and reduce your scientific images collected at the Rosemary Hill Observatory (RHO).


### Most Recent Version: November 21, 2025

**NOTE:** The data reduction team at RETRHO is currently working on the developement of new interactive features to do photometric calculations using the reduced data. Stay tuned.

## Features
### Data Reduction
* Classifies the Raw image frames into different sub-directories based on the frame type of the images stored in the `HEADER` of the `.fits` files (see `./data_reduction_codes/sort_observations.py`).
* Automatically reduces scientific frames by classifying Light frames based on the individual observed objects using the corresponiding callibration frames (see `data_reduction_codes/image_reduction.py`). These are the steps taken during the data reduction process:
    * Create master bias frame
    * Create master dark frames
    * Create normalize master flat frames
    * Reduce light frames using the corresponding master dark and normalized master flat
    * Remove bad/hot pixels
    * Subtract sky background (*optional feature*)
    * Align the reduced images (See the documentation on how to use the interactive alignment tool [here](./data_reduction_codes/image_alignment_instructions.md))
* Generates a detailed data reduction log and outlines all the image reduction steps for every input file.

### PSF Photometry Calculations
This tool was build to interactively perform Point Spread Function (PSF) calculations on a selected group of stars from the reduced frames. We recommend using this tool for timeseries analysis, particularly focusing on exoplanet transits, variable stars, and/or transients. For this tool the user will:
* Interactively select 1 or more stars to measure photometry for. Ideally the first star that the user selects is the main observed target, and the other stars are the reference stars to do a relative photometry analysis.
* Automatically, the tool will perform PSF photometry for the selected stars.
* The output is a dataframe containing:
    * The File name of the reduce frame
    * The measured Barycentric Julian Day (BJD) extracted from the observing date and time, and the telescope coordinates
    * The measured flux (PSF area sum) for each star and the associated uncertainty.
    * The converted instrumental magnitude ($M_{inst}$) from the meaured flux for each star and the associated uncertainty
    * The (X, Y) pixel coordinate of the brightest pixel around each star.

The scrip for tool can be found in `./photometry_analysis/psf_photometry.py`, and the instructions on how to use it can be found [here](https://github.com/explorerjs32/rho_data_reduction_pipeline/blob/main/photometry_analysis/psf_photometry_instructions.md). 

We also have a Jupyter Notebook that explains how to visualize and analyze the PSF photometry results, which can be found [here](https://github.com/explorerjs32/rho_data_reduction_pipeline/blob/main/photometry_analysis/psf_photometry_analysis.ipynb).

## Dependencies
The scripts that run this pipeline were developed using Python 3.8.20, which can be installed [Here](https://anaconda.org/anaconda/python/files?page=0&sort=distribution_type&sort_order=asc&version=3.8.20).

You may also download the most recent version of anaconda from [this link](https://www.anaconda.com/download) and then create an environment with the respective Python version. To create the environment you can use the following command:

`conda create --name <myenv> python=3.8`

and then activate it by running the following command:
`conda activate <myenv>`

Additionally, the following Python libraries were used along with their respective versions:
* `astroalign: 2.5.2`
* `astropy: 5.1`
* `numpy: 1.24.3`
* `matplotlib: 3.7.5`
* `pandas: 2.0.3`
* `photutils: 1.8.0`
* `scipy: 1.10.1`
* `tqdm: 4.67.1`
* `astroquery: 0.4.10`

If you need to install these specific versions of the libraries above, you can use the following command:
`conda install <package_name>=<version_number>`

## Installation
For non-git users you can download the RETRHO data reduction pipeline and other tools by clicking on the *green* "<> Code" button and click on "Download ZIP"

For git users you can clone the repository by typing on your terminal/command prompt the following command:

`git clone git@github.com:explorerjs32/rho_data_reduction_pipeline.git`

## Usage
### Data Reduction
The data reduction process is divided in two different steps: (1) sorting raw observations, and (2) reducing raw images for one or more objects. The codes to complete these steps can be found in `./data_reduction_codes/`. Let's look at what scripts to run and how to run them to complete these steps.

#### Sorting Raw Images
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



#### Reducing Raw Images
After the raw images have been classified into their different sub-directories. You can run the script `image_reduction.py` by parsing in these directories. This script can also be found in the `./data_reduction/` directory of this repository.

Similarly to the `sort_observations.py` script, you will be required to parse in different in order to get the expected outcome. These arguments are:

* `-l`: Directory containing the raw light frames from an object (e.g. `<raw_data_dir>/Light/<obj_name>`)
* `-d`: Directory containing the dark frames (e.g. `<raw_data_dir>/Dark/`)
* `-f`: Dierctory containing the flat frames (e.g. `<raw_data_dir>/Flat/`)
* `-b`: Directory containing the bias frames (e.g. `<raw_data_dir>/Bias/`)
* `-B`: *(optional boolean)* `True` or `False` argument to perform sky background subtraction. Default value is set to `True`
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

#### Interactive Image Reduction
After the raw images have been classified into their different sub-directories. You can run the script `image_reduction_interact_select.py` . This script can also be found in the `./data_reduction/` directory of this repository.

This script functionally operates the same as `image_reduction.py` described above, but rather than manually specifying the directory paths, a GUI window will appear, allowing you to interactively select the paths within your finder or file explorer to the calibration and object frames. There is currently an option to add additional light frame directories for reducing multiple objects with the same calibration data, as well as options to add additional dark or flat frame directories if you need to use calibration frames taken separately from your main observations. 

The notes related to the existing raw image reduction functions apply to this script as well. 

## Future Implementations
The RETRHO data reduction team is currently working on developing different interative tools to do photometric calculations or generate color images from the reduced frames. 

## Acknowledgments
This code has been developed by the RETRHO data reduction team. The team would like to appreciate the contributions from current and previous members for helping developing these tools.

### Current and Former Contributors
* Francisco Mendez
* Zabdiel Sanchez
* Ben Capistrant
* Jackson Lyle
* Georgeanne Johnson
* Santiago Roa
* Stefano Candiani
* Cassidy Camera
* Hannah Luft
* Leslie Morales
* Daniel Acosta
