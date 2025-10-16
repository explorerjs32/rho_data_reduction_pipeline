# PSF Photometry Tool
## Overview
The PSF Photometry Tool is designed for astronomical image analysis, providing precise photometry measurements of stars across multiple images. It features an interactive interface for star selection and automatically calculates flux and instrumental magnitudes with robust uncertainty estimates.

## Data Preparation
Before using the tool, ensure you have:

1. A directory containing reduced FITS images
2. A `frame_info.csv` file in the directory with space-separated columns:
    - `File`: File name of each image
    - `Object`: Target object name
    - `Exptime`: Exposure time in seconds
    - `Filter`: Filter used for observation
    - `Optional`: DATE-OBS, RA, DEC for BJD calculation
3. An `uncertainties.csv` file in the directory with space-separated columns:
    - `Read_Noise`: Detector read noise
    - `Dark_Current_<exptime>s`: Dark current for specific exposure times
    - `Flat_<filter>_Noise`: Flat field noise for specific filters

## Usage
Run the tool from the command line:

`python psf_photometry.py -d </path/to/data_directory/>`

**Interactive Star Selection:**

1. When launched, the tool displays the first image in the dataset
2. Use the mouse to draw selection rectangles around stars:
    - Click and drag to create a rectangle
    - The tool automatically finds the brightest pixel in the selection
    - Selected stars are marked with a red X and numbered
3. Select as many stars as needed for your analysis
4. Click the "Done with Star Selection" button when finished

**Output:**

Results are saved in a CSV file in a sub-directory inside the data directory that was parsed into the script. This CSV file contains:
- `File`: Image filename
- `BJD`: Barycentric Julian Date
- `Flux_Star_N`: Flux measurement for star N
- `Flux_err_Star_N`: Flux uncertainty for star N
- `Minst_Star_N`: Instrumental magnitude for star N
- `Minst_err_Star_N`: Magnitude uncertainty for star N
- `Star_N_x`, `Star_N_y`: Pixel coordinates of star N

## Photometry Algorithm
The PSF photometry process includes:

1. **Star Detection:** User-selected regions are analyzed to find precise star centers
2. **Background Estimation:** Local background is calculated using sigma-clipped statistics
3. **Source Extraction:** Connected pixel regions above threshold are identified
4. **Flux Measurement:** Total flux is measured within the identified stellar region
5. **Uncertainty Calculation:** Error propagation accounts for all noise sources

## Uncertainty Calculations
**Flux Uncertainty:**

The flux uncertainty $N_{*}$ is calculated by combining multiple noise sources in quadrature:

$N_{*} = \sqrt{GF_{*} + (n_{pix}GF_{D}) + (n_{pix}GF_{F}) + (n_{pix}(GF_{R})^2)}$

Where:
- $G$ is the Gain of the CCD (0.37 [e/ADU])
- $F_{*}$ is the measured star flux in [ADU/pixel]
- $n_{pix}$ is the number of pixels inside the PSF measuring star flux
- $F_{D}$ is the measured dark current in [ADU/pixel]
- $F_{F}$ is the measured flat frame noise in [ADU/pixel]
- $F_{R}$ is the read noise in [ADU/pixel]

## Instrumental Magnitude Uncertainty
The instrumental magnitude uncertainty is derived from the flux uncertainty through error propagation from the magnitude equation:

$m = -2.5 \log (F_{*}G / t)$

Its associated uncertainty is then given by:

$\sigma_{inst} = (2.5 / \ln 10) * (\sigma_{F_{*}} / F_{*}G)$

The factor $2.5 / log(10) ≈ 1.0857$ represents the propagation of relative flux error to magnitude units.

## Best Practices
1. **Star Selection:** Choose isolated stars with good signal-to-noise ratios
2. **Reference Stars:** Include non-variable stars for differential photometry
3. **Uncertainty Handling:** Pay attention to error bars when analyzing light curves

## Recomended Targets
The PSF photometry tool will have a best performance for:
- Exoplanet Transits
- Variable Stars

The user could also use it for photometry in cluster stars, but the zero-point callibrations will have to be done separately.

## References
- Photometric error calculation follows the methodology described in [Collins et al. 2017](https://iopscience.iop.org/article/10.3847/1538-3881/153/2/77).