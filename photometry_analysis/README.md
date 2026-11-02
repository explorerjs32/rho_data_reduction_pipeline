# Photometry Analysis Tools Documentation

## Table of Contents
- [Overview of Photometry Analysis](#overview-of-photometry-analysis)
- [Common Data Preparation](#common-data-preparation)
- <details><summary><a href="#1-aperture-photometry-tool">1. Aperture Photometry Tool</a></summary>

    - [Overview](#overview)
    - [Usage](#usage)
        - [Terminal Binning Setup](#terminal-binning-setup)
        - [Interactive Reference Image Selection](#interactive-reference-image-selection)
        - [Interactive Aperture Photometry Tool Part 1](#interactive-aperture-photometry-tool-part-1)
        - [Interactive Aperture Photometry Tool Part 2](#interactive-aperture-photometry-tool-part-2)
    - [Output](#output)
    - [Photometry Algorithm](#photometry-algorithm)
    - [Uncertainty Calculations](#uncertainty-calculations)
    - [Best Practices](#best-practices)
    - [Recommended Targets](#recommended-targets)

    </details>

- <details> <summary><a href="#2-psf-photometry-tool">2. PSF Photometry Tool</a></summary>

    - [Overview](#overview-1)
    - [Running the Tool](#running-the-tool)
        - [Option 1: Interactive GUI Mode](#option-1-interactive-gui-mode)
        - [Option 2: Command-Line Mode](#option-2-command-line-mode)
    - [Interactive Workflow](#interactive-workflow)
        - [1. Terminal Binning Setup](#1-terminal-binning-setup-1)
        - [2. Star Selection (First Filter)](#2-star-selection-first-filter)
        - [3. Reference Tracking (Subsequent Filters)](#3-reference-tracking-subsequent-filters)
    - [Output](#output-1)
    - [Photometry Algorithm](#photometry-algorithm-1)
    - [Uncertainty Calculations](#uncertainty-calculations-1)
    - [Best Practices](#best-practices-1)
    - [Recommended Targets](#recommended-targets-1)

</details>

- [References](#references)

## Overview of Photometry Analysis
These tools are designed for astronomical image analysis, providing precise photometry measurements of stars across single or multiple images. They feature interactive interfaces for star selection and calculate flux and instrumental magnitudes with robust uncertainty estimates. 
We also have a Jupyter Notebook that explains how to visualize and analyze the PSF photometry results, which can be found [here](./psf_photometry_analysis.ipynb).

## Common Data Preparation
Before using either tool, ensure you have:
1. A directory containing reduced FITS images.
2. A `frame_info.csv` file in the directory with space-separated columns:
    - `File`: File name of each image
    - `Object`: Target object name
    - `Exptime`: Exposure time in seconds
    - `Filter`: Filter used for observation
    - *Optional (For PSF Tool)*: DATE-OBS, RA, DEC for BJD calculation
3. An `uncertainties.csv` file in the directory with space-separated columns (written by the image reduction scripts):
    - `Gain`: Detector gain [e-/ADU]
    - `Read_Noise`: Detector read noise [ADU]
    - `Dark_Current_Rate`: Dark current [e-/pixel/s]; multiplied by the exposure time to get the dark signal of a frame
    - `Master_Dark_Noise_<exptime>s`: Uncertainty of each pixel of the master dark [ADU]
    - `Flat_<filter>_Noise`: Fractional uncertainty of each pixel of the normalized master flat
4. A `SKYSUB` header card in each reduced image: the sky level [ADU/pixel] removed by the background subtraction. Its photon noise is still in the image, so the photometry adds it back into the noise estimate.

---

## 1. Aperture Photometry Tool
### Overview 
The Aperture Photometry tool provides photometry measurements of stars across an individual combined image per filter. It features interactive interfaces for selecting a reference image and allows the user to select specific stars and a background to calculate flux.

### Usage 
Run the tool from the command line: `python aperture_photometry.py -d </path/to/data_directory/>`

#### Terminal Binning Setup
For each unique filter detected in your dataset, the terminal will prompt you to define a binning size.
- Enter 1 to combine all images into one bin.
- Enter the total number of images to ensure no binning.
- Enter any other valid integer to split images into groups. These images would then combine and represent a datapoint per bin. *Note: the tool will reject inputs that causes a bin to contain less than two images. Please select a different integer value that would ensure that each bin contains more than 2 images.*

#### Interactive Reference Image Selection
1. When launched, the first widget will display all combined images per filter.
2. Select one reference image in the checkbox that shows the brightest stars.
    - Click on checkbox.
    - Please only select one image, if you select multiple images, you can reclick the checkbox to unselect the image.
3. Click the Done button to proceed to the second widget.

#### Interactive Aperture Photometry Tool Part 1
1. Upon selecting a reference image from the previous widget, the second widget will display the reference image.
2. Use the mouse to draw selection rectangle around a star:
    - Click and drag to create a rectangle.
    - The tool automatically finds the brightest pixel in the selection.
3. Upon selecting a star, click Add Star:
    - This allows the tool to create an aperture around the star.
    - Clicking the - or + button under Add Star will allow the user to decrease or increase the aperture radius for the star.
    - Ensure the aperture covers the star without including the background.
    - If the user selects the wrong star, they are able to click the red undo button next to the + button to unselect the star.
4. Repeat steps 2 and 3 for as many stars needed for aperture photometry:
    - If the user would like to do differential photometry, also include reference stars in your selection.
    - User will need to keep track of which stars are their primary targets and which ones will be reference stars.
5. After selecting all the stars, use the mouse to draw selection rectangle around an area of the image that contains no stars:
    - Click and drag to create a rectangle.
    - The tool automatically finds the brightest pixel in the selection.
6. Click Add Background:
    - This allows the tool to create an aperture around the background region.
    - Clicking the - or + button under Add Background will allow the user to decrease or increase the aperture radius for the background.
    - Ensure the aperture does not contain any stars.
    - If the user does not like the region selected they can click the red undo button next to the + button to unselect the background.
7. Upon having selected both stars and background in the image, click the Aperture Photometry button:
    - This allows the tool to conduct aperture photometry on the reference image.
    - The output of the photometry will be displayed in the terminal.
8. If the user is satisfied with the photometry results, click Done.

#### Interactive Aperture Photometry Tool Part 2
1. Upon conducting aperture photometry on the reference image, the third widget will display two images:
    - The image on the left will be the reference image with the stars and background region displayed.
    - The image on the right will be another image from a different filter that will contain the background region chosen from the reference image.
2. The user will reselect the same stars in the same order from what is displayed in the reference image:
    - Use the mouse to draw selective rectangle around the star.
    - Click Add Star to create the aperture.
    - Click - or + to decrease or increase the aperture radius around the star and ensuring the aperture covers the star.
    - If the user selects the incorrect star, click the red undo button.
3. If the user has multiple other frames to do photometry, they can click the Next Filter or Previous Filter buttons to move between frames.
4. Repeat step 2.
5. Upon having selected all the stars across all images, click Aperture Photometry:
    - This enables the tool to conduct aperture photometry on all other frames.
    - Photometry tables per filter will be displayed in the terminal.
6. If the user is satisfied with the photometry tables, click Done.

### Output
Results are saved individually in a CSV file per filter inside the data directory that was parsed into the script. This CSV file contains:
- `File`: Image filename of one of the images that were used for the bin.
- `X_Center_Star_N`: The x pixel position for the Star. 
- `Y_Center_Star_N`: The y pixel position for the Star. 
- `Radius_Star_N`: The radius of the aperture for the Star. 
- `Net_Aperture_Sum_Star_N`: Aperture sum for the Star.
- `Net_Aperture_Sum_Error_Star_N`: Aperture sum error for the Star
- `Minst_Star_N`: Instrumental magnitude for the Star
- `Minst_Error_Star_N`: Instrumental magnitude error for the Star

The CSV file also contains the background information for the region selected.

### Photometry Algorithm
The Aperture photometry process includes:
1. **Star Detection:** User-selected regions are analyzed to find precise star centers
2. **Background Selection:** User-selected background for background subtraction. 
3. **Aperture Sum Measurement:** Aperture sum is measured per star within the aperture selected by the user. 
4. **Uncertainty Calculation:** Error propagation accounts for all noise sources

### Uncertainty Calculations
**Aperture Sum Uncertainty:**
The aperture sum uncertainty $N_{*}$ follows the CCD equation (Collins et al. 2017, Appendix B, Eq. 7), extended for bins of $N$ co-added images and for the noise of the master dark and flat. All quantities are in electrons:

$N_{*} = \sqrt{F_{*} + n_{pix} (1 + n_{pix}/n_{back}) (S + N (D + R^2)) + n_{pix} N^2 \sigma_{MD}^2 + \sigma_f^2 \sum_{ap} P_i^2}$

Where:
- $F_{*}$ is the net (sky-subtracted) star flux [e-], measured on the co-added image
- $n_{pix}$ is the number of pixels inside the aperture measuring star flux
- $n_{back}$ is the number of pixels inside the aperture measuring background flux
- $S$ is the sky per pixel [e-/pixel] of the co-added image: the residual measured in the background aperture plus the sum of the `SKYSUB` levels removed by the reduction
- $N$ is the number of images co-added into the bin
- $D$ is the dark signal of one frame [e-/pixel] (`Dark_Current_Rate` × exposure time). Dark subtraction removes the mean dark signal but not its Poisson noise
- $R$ is the read noise [e-] (`Read_Noise` × gain)
- $\sigma_{MD}$ is the uncertainty of a pixel of the master dark [e-] (`Master_Dark_Noise_<exptime>s` × gain)
- $\sigma_f$ is the fractional flat-field uncertainty and $P_i$ the pixel values [e-] inside the aperture

The master dark and flat are the same in every frame, so their errors add up coherently in a sum of frames: the master dark term grows as $N^2$ rather than $N$, and the flat term scales with the summed signal. This is exact when the stars stay on the same detector pixels (minimal drift) and an upper limit otherwise. For co-added bins, the master dark noise stays below the read noise contribution as long as the number of dark frames is at least about 1.6 × the number of light frames per bin.

**Instrumental Magnitude Uncertainty:**
The instrumental magnitude uncertainty is derived from the flux uncertainty through error propagation from the magnitude equation, with the total exposure time of the co-added images:

$m = -2.5 \log (F_{*} / (N t))$

Its associated uncertainty is then given by:
$\sigma_{inst} = (2.5 / \ln 10) * (N_{*} / F_{*})$

The factor $2.5 / \log(10) \approx 1.0857$ represents the propagation of relative flux error to magnitude units.

### Best Practices
1. **Binning Setup:** Choose an appropriate bin setup to ensure a consistent number of images per bin
2. **Star Selection:** Choose stars that are isolated from their companions
3. **Star Aperture Size:** Ensure that the aperture properly covers the star and does not include background
4. **Reference Stars:** Include non-variable stars for differential photometry
5. **Background Aperture Size:** Ensure that the aperture does not include any stars

### Recommended Targets
The aperture photometry tool will have a best performance for:
- Active Galactic Nuclei 
- Clusters
- Supernovae

If the user wants zeropoint and apparent magnitude calculations, this will need to be done separately.

---

## 2. PSF Photometry Tool

### Overview
The PSF Photometry Tool provides precise photometry measurements of stars across multiple images. It features an interactive interface for star selection and automatically calculates flux and instrumental magnitudes with robust uncertainty estimates.

### Running the Tool
The tool supports dual-input modes for convenience: command-line execution or interactive GUI selection. 

#### Option 1: Interactive GUI Mode
Run the script without any arguments:
`python psf_photometry_binning.py`

A popup window will appear allowing you to visually browse and select your "Reduced data directory" and "Output directory". 

#### Option 2: Command-Line Mode
Bypass the GUI entirely by parsing your directories directly in the terminal:
`python psf_photometry_binning.py -d /path/to/data/ -o /path/to/output/`

### Interactive Workflow
Once the directories are selected, the tool processes your data filter by filter. 

#### 1. Terminal Binning Setup
For each unique filter detected in your dataset, the terminal will prompt you to define the binning size: 
- Enter 1 to co-add all images into a single master bin. 
- Enter the total number of images to proceed with no binning (1 image per bin). 
- Enter any other valid integer to split the images into groups. *Note: The tool will reject inputs that leave any bin with only 1 image (unless you are explicitly requesting 1 image per bin across the board).* 

#### 2. Star Selection (First Filter)
- A single-panel plot will display the first image (or first binned image) of the first filter. 
- Click and drag to create selection rectangles around your target stars. 
- The tool automatically finds the brightest pixel and marks it with a red X and a sequential label (e.g., Star 1, Star 2). 
- Click Done with Star Selection when finished. 

#### 3. Reference Tracking (Subsequent Filters)
- For all subsequent filters, the visualization tool will launch a dual-panel stacked plot. 
- **Top Plot:** Displays the image from your first filter, keeping your previously selected stars and labels visible as a reference guide. 
- **Bottom Plot:** Displays the current filter's image. 
- Select the stars on the bottom plot in the exact same order as the reference plot above it to ensure consistency. 

### Output
Results are saved in a CSV file in a sub-directory inside the data directory that was parsed into the script. This CSV file contains:
- `File`: Image filename
- `BJD`: Barycentric Julian Date
- `Flux_Star_N`: Flux measurement for star N
- `Flux_err_Star_N`: Flux uncertainty for star N
- `Minst_Star_N`: Instrumental magnitude for star N
- `Minst_err_Star_N`: Magnitude uncertainty for star N
- `Star_N_x`, `Star_N_y`: Pixel coordinates of star N

### Photometry Algorithm
The PSF photometry process includes:
1. **Star Detection:** User-selected regions are analyzed to find precise star centers
2. **Background Estimation:** Local background is calculated using sigma-clipped statistics
3. **Source Extraction:** Connected pixel regions above threshold are identified
4. **Flux Measurement:** Total flux is measured within the identified stellar region
5. **Uncertainty Calculation:** Error propagation accounts for all noise sources

### Uncertainty Calculations
**Flux Uncertainty:**
The flux uncertainty $N_{*}$ is calculated by combining multiple noise sources in quadrature. All quantities are in electrons:

$N_{*} = \sqrt{F_{*} + n_{eff} S + n_{eff} N (D + R^2) + n_{eff} N^2 \sigma_{MD}^2 + (\sigma_f F_{*})^2 / n_{eff}}$

Where:
- $F_{*}$ is the star flux [e-] (volume of the fitted Gaussian × gain), measured on the co-added image
- $n_{eff} = 4 \pi \sigma_x \sigma_y$ is the effective number of pixels of the fitted PSF
- $S$ is the sky per pixel [e-/pixel] of the co-added image: the sum of the `SKYSUB` levels removed by the reduction plus any residual sky fitted by the model
- $N$ is the number of images co-added into the bin
- $D$ is the dark signal of one frame [e-/pixel] (`Dark_Current_Rate` × exposure time)
- $R$ is the read noise [e-] (`Read_Noise` × gain)
- $\sigma_{MD}$ is the uncertainty of a pixel of the master dark [e-] (`Master_Dark_Noise_<exptime>s` × gain); it grows as $N^2$ because the same master dark is subtracted from every frame
- $\sigma_f$ is the fractional flat-field uncertainty; for a Gaussian PSF, $\sum (\sigma_f P_i)^2 = (\sigma_f F_{*})^2 / n_{eff}$, where $P_i$ are the pixel values [e-]

**Instrumental Magnitude Uncertainty:**
The instrumental magnitude uncertainty is derived from the flux uncertainty through error propagation from the magnitude equation, with the total exposure time of the co-added images:

$m = -2.5 \log (F_{*} / (N t))$

Its associated uncertainty is then given by:
$\sigma_{inst} = (2.5 / \ln 10) * (N_{*} / F_{*})$

The factor $2.5 / \log(10) \approx 1.0857$ represents the propagation of relative flux error to magnitude units.

### Best Practices
1. **Star Selection:** Choose isolated stars with good signal-to-noise ratios
2. **Reference Stars:** Include non-variable stars for differential photometry
3. **Uncertainty Handling:** Pay attention to error bars when analyzing light curves

### Recommended Targets
The PSF photometry tool will have a best performance for:
- Exoplanet Transits
- Variable Stars

The user could also use it for photometry in cluster stars, but the zero-point calibrations will have to be done separately.

---

## References
- Photometric error calculation follows the methodology described in [Collins et al. 2017](https://iopscience.iop.org/article/10.3847/1538-3881/153/2/77).