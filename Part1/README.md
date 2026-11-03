# Code map
- `Task1.py`: main fruit detection demo; defaults to `darkFactor=0.8`, `mirror=True`.
- `Task1_functions.py`: colour filtering, mask cleanup, geometry and plotting helpers.
- `WhatIf_Task1.py`: 19 brightness levels, detection CSV, annotated images and overview.
- `Task1_Apple_DiffStage_Image.py`: apple filtering-stage comparisons.
- `Task1_Whatif_DataAnalysis.m`: MATLAB four-panel analysis of the what-if CSV.

AI use: GPT6 Astra

## Included files and results

- `Fruits.jpg`: input image used by the Python scripts.
- `Task1_Whatif_Results/`: detection CSV, 19 annotated images, overview PNG/PDF and `Task1_MatlabPlot.png`.
- `Task1_Apple_Stage_Results/`: supplied apple processing-stage images.

## Run the Python scripts

Install dependencies:
```bash
python -m pip install numpy pandas matplotlib
```
From the repository root:
```bash
python Part1/Task1.py
python Part1/WhatIf_Task1.py
python Part1/Task1_Apple_DiffStage_Image.py
```
Paths are resolved from each script, so these scripts also work from other working directories.
`Task1.py` displays the default viva image. `WhatIf_Task1.py` uses `mirror=False`
and sweeps dark factors from 1.00 to 0.10 in steps of 0.05, writing 57 records
and the figures to `Task1_Whatif_Results/`.

The apple-stage script currently writes regenerated images to `apple_stage_results/`;
the supplied copies are stored in `Task1_Apple_Stage_Results/`.

## MATLAB analysis

After generating the what-if CSV, open and run `Task1_Whatif_DataAnalysis.m` in MATLAB.
It reads `Task1_Whatif_Results/fruit_detection_data_mirror_False.csv` and displays
four panels: bounding-box width, height, centroid distance and area.
The script displays the figure; it does not automatically save it. Use MATLAB's
figure export to update `Task1_Whatif_Results/Task1_MatlabPlot.png`.
MATLAB is required only for this analysis figure.

Box area is geometric width times height. It is not the visible-fruit coverage
measure, and the CSV's `detected` flag records whether a bounding box exists.
