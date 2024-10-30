# Data Reduction Pipeline

This project provides a data reduction pipeline for processing astronomical data stored in FITS files. The pipeline extracts relevant information from FITS file headers and compiles it into pandas DataFrames for further analysis.

## Features

- Extracts object names, frame types, exposure times, and filters from FITS file headers.
- Generates a detailed observing log based on the extracted frame information.

## Usage

To use the data reduction pipeline, you need to specify the directory where your collected FITS data is stored. You can also optionally specify directories for bias, dark, flat, and light frames. 

### Command Line Arguments

- `-D` or `--data` (required): Directory where the collected data is stored.
- `-b` or `-bias_frames`: Directory where the bias frames are stored.
- `-d` or `--dark_frames`: Directory where the dark frames are stored.
- `-f` or `--flat_frames`: Directory where the flat frames are stored.
- `-l` or `--light_frames`: Directory where the light (science) frames are stored.

For example, when running from the command line, in the rho_data_reduction_pipeline folder:

python data_reduction_codes/image_reduction.py -data data/2024-04-15/

NOTE: you must add the "/" at the end of the directory specifying where the data is located, or the program will not be able to navigate inside that folder 
