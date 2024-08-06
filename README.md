# Assesment of global impact of circular interventions wihtn Input output tablles 
This study aimed to assess the global impact of circular interventions by integrating these interventions into the Exiobase database (both hybrid and monetary tables). The repository provides all scripts utilized in this study and just the function to implement the changes. This can be utilized in own studies regarding modelling interventions. However the scripts that are used to finilise the thesis are noted down below. Only use the following scripts as representative scripts for the thesis.

Data utilized in the code can be downloaded from Zenodo using the following links.
Monetary tables (MRIOT) 
- https://zenodo.org/records/5589597
- IOT_2011_ixi.zip
  
Hybrid tables 
- https://zenodo.org/records/10148587
- MR_HIOT_2011_v3_3_18_by_product_technology.csv
- MR_HIOT_2011_v3_3_18_extensions.xlsx

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
- Requires Sequence of schocks included in each scenario. (done by using list of index values)
- Vector containing all environmental impact of interest
- Vector containing the environmental sensitivity
- Sensitivity input to modify the inputs in the shocks
- Threshold (integer) and indicator (string) containing input parameters for visuals (graphs)

Application of this function can be found in both [monetary tables](function_monetary.py) and [hybrid tables](function_hybrid.py)
 
