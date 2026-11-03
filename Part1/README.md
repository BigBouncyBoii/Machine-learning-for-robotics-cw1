# Code map
- Task1.py: default viva image (darkFactor=0.8, mirror=True).
- Task1_functions.py: shared image processing functions.
- WhatIf_Task1.py: brightness experiment, CSV, individual images and overview.
- Task1_Whatif_DataAnalysis.m: four-panel MATLAB analysis.
- Task1_Apple_DiffStage_Image.py: apple processing-stage comparisons.

AI use: GPT6 Astra

Install dependencies: `python -m pip install numpy pandas matplotlib`

Run from this directory:
```
python Task1.py
python WhatIf_Task1.py
python Task1_Apple_DiffStage_Image.py
```
Paths are resolved relative to each script; Fruits.jpg is included.
Results are saved in Task1_Whatif_Results and apple_stage_results.
Box area is geometric width times height, not a visible-fruit coverage measure.

Run Task1_Whatif_DataAnalysis.m in MATLAB after generating the what-if CSV.
The MATLAB figure is provided as Task1_Whatif_Results/Task1_MatlabPlot.png.
