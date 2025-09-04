# RETRHO Data Reduction Pipeline

Welcome to the RETRHO data reduction pipeline repository! Here, you will find the necesary tools to automatically callibrate and reduce your scientific images collected at the Rosemary Hill Observatory (RHO).

### Most Recent Version: September 3rd, 2025

**NOTE:** The data reduction team at RETRHO is currently working on the developement of new interactive features to do photometric calculations using the reduced data. Stay tuned.

## Features
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

If you need to install these specific versions of the libraries above, you can use the following command:
`conda install <package_name>=<version_number>`

## Installation
For non-git users you can download the RETRHO data reduction pipeline and other tools by clicking on the *green* "<> Code" button and click on "Download ZIP"

For git users you can clone the repository by typing on your terminal/command prompt the following command:

`git clone git@github.com:explorerjs32/rho_data_reduction_pipeline.git`

## Usage
### Data Reduction
The data reduction process is divided in two different steps: (1) sorting raw observations, and (2) reducing raw images for one or more objects. Thw codes to complete these steps can be found in `./data_reduction_codes/`. Let's look at what scripts to run and how to run them to complete these steps.

#### Sorting Raw Images
Once you have downloaded the raw images from RHO, you can parse in the data directory to into the `sort_observations.py` script. This script takes in a single argument called `--dir` or `-d` for short, which should point to the directory that has the raw images. 

You can run the script as follows:

`python sort_observations.py -d <path_to_raw_data>`

**Note:** You should parse in a directory rather than a list of files or a single file for the code to work properly.

The output of this directory will be a copy of the raw image files inside the parsed directory, but they will be re-organized into four different sub-directories based on the frame type of the image (Light, Dark, Flat, and Bias). The light frames sub-directory will also stored the re-organized frames by the object name, and there will be an additional sub-directory for each object individually.

These sub-directories will be usefull for the next step, which is the image reduction process.

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

## Future Implementations
The RETRHO data reduction team is currently working on developing different interative tools to do photometric calculations or generate color images from the reduced frames. 

## Acknowledgments
This code has been developed by the RETRHO data reduction team. The team would like to appreciate the contributions from current and previous members for helping developing these tools.

### Current and Former Team Members
* Francisco Mendez
* Hannah Luft
* Stefano Candiani
* Cassidy Camera
* Leslie Morales
* Ben Capistrant
* Santiago Roa
* Zabdiel Sanchez
