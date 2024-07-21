# Assesment of global impact of circular interventions wihtn Input output tablles 
This study aimed to assess the global impact of circular interventions by integrating these interventions into the Exiobase database (both hybrid and monetary tables). The code provided in this repository. The repository provides all scripts utilized in this study and just the function to implement the changes. This can be utilized in own studies regarding modelling interventions.  

## Scripts used in the study to perform projections
- [perform projections on monetary tables](baseline2050.py)
- [perform projections on hybrid tables](baselinehybrid2050.py)
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

