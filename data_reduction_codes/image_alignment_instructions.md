## Interactive Manual Alignment Tool

The Interactive Manual Alignment Tool provides a graphical interface for aligning astronomical images when automatic alignment fails. This tool is essential for ensuring high-quality data reduction, especially in cases where images have large offsets, low signal-to-noise, or artifacts that prevent automated routines from working reliably.

---

### When Is This Tool Triggered?

The tool is automatically invoked by the pipeline when the standard image alignment (typically using `astroalign` or similar algorithms) cannot successfully register a target image to the template. This may occur due to:

- Insufficient or ambiguous features for matching (e.g., few stars, cosmic rays, or artifacts).
- Large shifts or rotations between images.
- Unusual image distortions or defects.

When such a scenario is detected, the pipeline launches the Interactive Manual Alignment Tool, pausing automated processing and allowing the user to intervene.

---

### User Interface Overview

Upon activation, the tool displays a window with the following layout:

- **Top Left:** Template (reference) image.
- **Top Right:** Target (image to be aligned).
- **Bottom Left:** Overlay of template and target images for visual comparison.
- **Bottom Right:** Control panel with interactive buttons.

Each image panel supports zooming (mouse scroll wheel) and panning (matplotlib toolbar) for precise navigation.

---

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

---

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

---

### Workflow Summary

1. The tool opens automatically if automatic alignment fails.
2. Select corresponding points in both the template and target images.
3. Click **Align** to preview the alignment in the overlay.
4. If satisfied, click **Done** to save the aligned image.
5. If not, use **Reset** to try again, **Accept As Is** to keep the original, or **Ignore Image** to skip.
6. The pipeline continues with the next image or step.

---

### Tips for Effective Alignment

- Select at least two well-separated, easily identifiable features for best results.
- Use zoom and pan to improve accuracy when clicking on features.
- If the overlay looks misaligned after pressing **Align**, try resetting and selecting more accurate or additional points.

---

### Troubleshooting

- If the tool does not appear, ensure your environment supports PyQt5 and matplotlib interactive backends.
- If you accidentally close the window, rerun the pipeline step to trigger the tool again.

---

This tool ensures robust and user-friendly manual alignment, allowing you to process even the most challenging astronomical images with confidence