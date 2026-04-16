# Aperture Photometry Tool
## Overview
The Aperture Photometry tool is designed for astronomical image analysis, providing photometry measurements of stars across an individual combined image per filter. It features interactive interfraces for selecting a reference image and allows the user to select specific stars and a background to calculate flux and instrumental magnitudes with robust uncertainty estimates.

## Data Preparation
Before using the tool, ensure you have:

1. A directory containing reduced FITS images
2. A `frame_info.csv` file in the directory with space-separated columns:
    - `File`: File name of each image
    - `Object`: Target object name
    - `Exptime`: Exposure time in seconds
    - `Filter`: Filter used for observation
3. An `uncertainties.csv` file in the directory with space-separated columns:
    - `Read_Noise`: Detector read noise
    - `Dark_Current_<exptime>s`: Dark current for specific exposure times
    - `Flat_<filter>_Noise`: Flat field noise for specific filters

## Usage
Run the tool from the command line:

`python aperture_photometry.py -d </path/to/data_directory/>`

**Interactive Reference Image Selection**

1. When launched, the first widget will display all combined images per filter.
2. Select one reference image in the checkbox that shows the brightest stars.
    - Click on checkbox
    - Please only select one image, if you select multiple images, you can reclick the checkbox to unselect the image
3. Click the Done button to proceed to the second widget.

**Interactive Aperture Photometry Tool Part 1**
1. Upon selecting a reference image from the previous widget, the second widget will display the reference image.
2. Use the mouse to draw selection rectangle around a star:
    - Click and drag to create a rectangle
    - The tool automatically finds the brightest pixel in the selection 
3. Upon selecting a star, click Add Star:
    - This allows the tool to create an aperture around the star.
    - Clicking the - or + button under Add Star will allow the user to decrease or increase the aperture radius for the star.
    - Ensure the aperture covers the star without including the background.
    - If the user selects the wrong star, they are able to click the red undo button next to the + button to unselect the star.
4. Repeat steps 2 and 3 for as many stars needed for aperture photometry:
    - If the user would like to do differential photometry, also include reference stars in your selection.
        - User will need to keep track of which stars are their primary targets and which ones will be reference stars
5. After selecting all the stars, use the mouse to draw selection rectangle around an area of the image that contains no stars:
    - Click and drag to create a rectangle
    - The tool automatically finds the brightest pixel in the selection
6. Click Add Background:
    - This allows the tool to create an aperture around the background region
    - Clicking the - or + button under Add Background will allow the user to decrease or increase the aperture radius for the background
    - Ensure the aperture does not contain any stars.
    - If the user does not like the region selected they can click the red undo button next to the + button to unselect the background
7. Upon having selected both stars and background in the image, click the Aperture Photometry button:
    - This allows the tool to conduct aperture photometry on the reference image.
    - The output of the photometry will be displayed in the terminal.
8. If the user is satisfied with the photometry results, click Done.

**Interactive Aperture Photometry Tool Part2**

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

**Output:**

Results are saved individually in a CSV file per filter inside the data directory that was parsed into the script. This CSV file contains:
- `Star` : Star number. 
- `X_Center`: The x pixel position for the Star. 
- `Y_Center`: The y pixel position for the Star. 
- `Radius`: The radius of the aperture for the Star. 
- `Net_Aperture_Sum`: Aperture sum for the Star.
- `Net_Aperture_Sum_Error`: Aperture sum error for the Star
- `Minst`: Instrumental magnitude for the Star
- `Minst_Error`: Instrumental magnitude error for the Star
The CSV file also contains a row with the background.

## Photometry Algorithm
The Aperture photometry process includes:

1. **Star Detection:** User-selected regions are analyzed to find precise star centers
2. **Background Selection:** User-selected background for background subtraction. 
3. **Aperture Sum Measurement:** Aperture sum is measured per star within the aperture selected by the user. 
4. **Uncertainty Calculation:** Error propagation accounts for all noise sources

## Uncertainty Calculations
**Aperture Sum Uncertainty:**

The aperture sum uncertainty $N_{*}$ is calculated by combining multiple noise sources in quadrature:

$N_{*} = \sqrt((F_{*_net}) + n_{pix} * (1 + (n_{pix}/{n_back})) * ((sky_{per_pixel}) + F_{D_adu}*gain + (F_{R_adu}*gain)**2 + (F_{flat_adu}*gain)**2))$

Where:
- $gain$ is the Gain of the CCD (0.37 [e/ADU])
- $F_{*_net}$ is the measured star flux in [ADU/pixel]
- $n_{pix}$ is the number of pixels inside the aperture measuring star flux
- $n_{back}$ is the number of pixels inside the aperture measuring background flux
- $sky_{per_pixel}$ is the per-pixel sky background in [e/pixel]
- $F_{D_adu}$ is the measured dark current in [ADU/pixel]
- $F_{flat_adu}$ is the measured flat frame noise in [ADU/pixel]
- $F_{R_adu}$ is the read noise in [ADU/pixel]

## Instrumental Magnitude Uncertainty
The instrumental magnitude uncertainty is derived from the flux uncertainty through error propagation from the magnitude equation:

$m = -2.5 \log (aperture_sum / t)$

Its associated uncertainty is then given by:

$\sigma_{inst} = (2.5 / \ln 10) * (N_{*} / aperture_sum)$

The factor $2.5 / log(10) ≈ 1.0857$ represents the propagation of relative flux error to magnitude units.

## Best Practices
1. **Star Selection:** Choose stars that are isolated from their companions
2. **Star Aperture Size** Ensure that the aperture properly covers the star and does not include background
3. **Reference Stars:** Include non-variable stars for differential photometry
4. **Background Aperture Size** Ensure that the aperture does not include any stars

## Recomended Targets
The aperture  photometry tool will have a best performance for:
- Active Galactic Nuclei 
- Clusters
- Supernovae

If the user wants zeropoint and apparent magnitude calculations, this will need to be done separately.

## References
- Photometric error calculation follows the methodology described in [Collins et al. 2017](https://iopscience.iop.org/article/10.3847/1538-3881/153/2/77).