# RETRHO Data Reduction Pipeline

Welcome to the RETRHO data reduction pipeline repository! Here, you will find the necesary tools to automatically callibrate and reduce your scientific images collected at the Rosemary Hill Observatory (RHO).


### Most Recent Version: July 21, 2026

**NOTE:** The data reduction team at RETRHO is actively working on the developement of new interactive features. Any feedback on how to improve the current tools or build new ones is welcome!


## Features and Tools
- [Image Reduction](./data_reduction_codes/README.md#image-reduction)
    - [Interactive Image Reduction](./data_reduction_codes/README.md#interactive-image-reduction)
    - [Interactive Image Alignment](./data_reduction_codes/README.md#interactive-manual-alignment-tool)
- [Photometry Analysis](./photometry_analysis/README.md)
    - [Aperture Photometry Tool](./photometry_analysis/README.md#1-aperture-photometry-tool)
    - [PSF Photometry Tool](./photometry_analysis/README.md#2-psf-photometry-tool)
    

## Installation
For non-git users you can download the RETRHO data reduction pipeline and other tools by clicking on the *green* "<> Code" button and click on "Download ZIP"

For git users you can clone the repository by typing on your terminal/command prompt the following command:

`git clone git@github.com:explorerjs32/rho_data_reduction_pipeline.git`

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
* `matplotlib: 3.7.3`
* `pandas: 2.0.3`
* `photutils: 1.8.0`
* `scipy: 1.10.1`
* `tqdm: 4.67.1`
* `astroquery: 0.4.7`
* `PyQt5: 5.15.11`

The following packages can be installed together via conda:

`conda install -c conda-forge astropy=5.1 matplotlib=3.7.3 pandas=2.0.3 scipy=1.10.1 tqdm=4.67.1`

The remaining packages are not available through conda and must be installed with pip:

`python -m pip install --no-user astroalign==2.5.2 numpy==1.24.3 photutils==1.8.0 astroquery==0.4.7 PyQt5==5.15.11`


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
* Andrea Moscoso
* Santiago Roa
* Stefano Candiani
* Cassidy Camera
* Hannah Luft
* Leslie Morales
* Daniel Acosta
