# Assesment of global impact of circular interventions wihtn Input output tablles 
This study aimed to assess the global impact of circular interventions by integrating these interventions into the Exiobase database (both hybrid and monetary tables). The repository provides all scripts utilized in this study and just the function to implement the changes. This can be utilized in own studies regarding modelling interventions. However the scripts that are used to finilise the thesis are noted down below. Only use the following scripts as representative scripts for the thesis.

The following scr

## Scripts utilised in this study
### Scripts used in the study to perform projections
- [Perform projections on monetary tables](baseline2050.py)
- [Perform projections on hybrid tables](baselinehybrid2050.py)
- [Visualise results from projections](visual_projections.py)

Both projections results are exported for visualisation in the last file 

### Scripts used to perform the shocks.
- [Perform analysis on monetary tables](function_monetary.py)
- [Perform analysis on monetary tables](function_hybrid.py)
- [Visualise results from projections(per scenario)](visual_stacked.py)
- [Visualise results from projections(per region)](visuals.py)

Both scenarios construction results are exported for visualisation in the last file.

### Pure funciton
- Function to perform adjustments to both A and Y matrix
- [Circular function](cirk_func.py)

Documentation regarding the function can be found here: [codedoc](api.html). The function can be seperatly used to perform the analysis. 
it contains the following variables that can be changed:
- Path to excel file containing all shocks 
- Requires both the baseline/base technical coeffiencient and final demand matrix
- Requires Sequence of schocks included in each scenario.
- Vector containing all environmental impact of interest
- Vector containing the environmental sensitivity
- Sensitivity input to modify the inputs in the shocks
- Threshold (integer) and indicator (string) containing input parameters for visuals (graphs)
 
