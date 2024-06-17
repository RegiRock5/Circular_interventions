# -*- coding: utf-8 -*-
"""
Created on Tue May 28 20:57:08 2024

@author: regin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
current_pop = 16655799
project_pop = 20610000
Populationgrowth = project_pop / current_pop
unit = "tonne"
#%% input path of the IOT 
iot_path = r"C:/Industrial_ecology/Thesis/IOT_2015_ixi"
save_path = r'C:/Industrial_ecology/Thesis/Circularinterventions/Code'
#%%Input of the HIOT 
hybrid_output_path = "C:/Industrial_ecology/Thesis/HIOT_2021_ixi"
# F_imp_hh = pd.read_csv(f'{hybrid_output_path}/impacts/F_Y.txt' , sep='\t', index_col=[0], header=[0, 1])

# Z_labels = pd.read_csv(f'{iot_path}/Z.txt', sep='\t', index_col=[0, 1], header=[0, 1])
# A = pd.read_csv(f'{iot_path}/A.txt', sep='\t', index_col=[0, 1], header=[0,1])
# Y = pd.read_csv(f'{iot_path}/Y.txt' , sep='\t', index_col=[0, 1], header=[0, 1])
# data input and modify
hiot_path = "C:/Industrial_ecology/Thesis/Circularinterventions/Data/"
Z_hybrid = pd.read_csv(f"{hiot_path}MR_HIOT_2011_v3_3_18_by_product_technology.csv", index_col=[0,1,2,3,4], header=[0,1,2,3])
Z_hybrid = pd.DataFrame(Z_hybrid.values, columns = Z_hybrid.columns, index = Z_hybrid.columns)
Z_hybrid = Z_hybrid.droplevel([2,3], axis=1).droplevel([2,3], axis=0) 
Y_hybrid = pd.read_csv(f"{hiot_path}MR_HIOT_2011_v3_3_18_FD.csv", index_col=[0,1,2,3,4], header=[0,1,2,3])
Y_hybrid = pd.DataFrame(Y_hybrid.values, columns = Y_hybrid.columns, index = Z_hybrid.columns)
Y_hybrid = Y_hybrid.droplevel([2,3], axis=1) 
Y_hybrid.NL = Y_hybrid.NL * Populationgrowth


#modify the final demand to project to 2050

x_hybrid = Z_hybrid.sum(axis = 1) + Y_hybrid.sum(axis = 1)
x_out = x_hybrid.copy()
x_out[x_out!=0] = 1/x_out[x_out!=0]
inv_diag_x = np.diag(x_out)
A_hybrid = Z_hybrid @ inv_diag_x
A_hybrid = pd.DataFrame(A_hybrid.values, columns = Z_hybrid.columns, index = Z_hybrid.columns)
I = np.eye(A_hybrid.shape[0])
L_hybrid = np.linalg.inv(I-A_hybrid)
# hybrid_output_path = "C:/Industrial_ecology/Thesis/HIOT_2021_ixi"

# A_hybrid.to_csv(f'{hybrid_output_path}/A.txt', sep='\t', index=True)  
# Y_hybrid.to_csv(f'{hybrid_output_path}/Y.txt',sep='\t', index=True)


#world_IOT = parse_exiobase_3(path = hybrid_output_path)

#%%
file_path = 'C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/shocks_full.xlsx'
sheet_name = 'z'  # Replace with the name of your sheet
Full_shocks_A = pd.read_excel(file_path, sheet_name=sheet_name)
print(Full_shocks_A)


#%% Implement shocks 
A_modify = A_hybrid.copy()
for _, row in Full_shocks_A.iterrows():
    country_row = row['row region']
    sector_row = row['row sector']
    country_column = row['column region']
    sector_column = row['column sector']
    value = row['value']
    typechange = row["type"]
    if typechange == "Percentage":
        A_modify.loc[(country_row, sector_row), (country_column, sector_column)] *= 1 + value
    else:
        A_modify.loc[(country_row, sector_row), (country_column, sector_column)] += value

        
#groupby to check results
A_modify1 = A_modify.sum(axis = 0)
sortedHybrid = A_modify1.groupby(level=0, axis=0, sort=False).sum()

A_hybrid = A_hybrid.sum(axis = 0)
sortedHybridBaseline = A_hybrid.groupby(level=0, axis=0, sort=False).sum()

diffchecker = pd.DataFrame()
diffchecker["baseline"] = sortedHybridBaseline
diffchecker["changes"] = sortedHybrid
diffchecker["diff"] = diffchecker["changes"] - diffchecker["baseline"]

A_modify.loc[("NL", "Cultivation of wheat"), ("NL", "Cultivation of wheat")]

    
#%%
file_path = 'C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/shocks_full.xlsx'
sheet_name = 'Y'  # Replace with the name of your sheet
Full_shocks_Y = pd.read_excel(file_path, sheet_name=sheet_name)
print(Full_shocks_Y)

Y_modify = Y_hybrid.copy()
for _, row in Full_shocks_Y.iterrows():
    country_row = row['row region']
    sector_row = row['row sector']
    country_column = row['column region']
    sector_column = row['demand category']
    value = row['value']
    Y_modify.loc[(country_row, sector_row), (country_column, sector_column)] *= 1 + value

#groupby to check results
Y_modify1 = Y_modify.sum(axis = 1)
YsortedHybrid = Y_modify1.groupby(level=0, axis=0, sort=False).sum()

Y_hybrid = Y_hybrid.sum(axis = 1)
YsortedHybridBaseline = Y_hybrid.groupby(level=0, axis=0, sort=False).sum()

diffchecker["baselineY"] = YsortedHybridBaseline
diffchecker["changesY"] = YsortedHybrid
diffchecker["diffY"] = diffchecker["changesY"] - diffchecker["baselineY"]

#%% Check difference between all by taking resource activity and emmissions (CO2)
extensions = pd.ExcelFile("C:/Industrial_ecology/Thesis/Circularinterventions/Data/MR_HIOT_2011_v3_3_18_extensions.xlsx")
extensions.sheet_names

resource = "Iron ores"
resource = "Bauxite and aluminium ores"
Emission = "Carbon dioxide, fossil"
#resource = "Copper ores"

#resource extraction --> take only the material of interest
RE = extensions.parse(sheet_name="resource_act", index_col=[0,1], header=[0,1,2,3]) 
RE_FD = extensions.parse(sheet_name="resource_FD", index_col=[0,1], header=[0,1,2,3]) 
RE = RE.loc[resource].sum(axis = 0)
RE_FD = RE_FD.loc[resource]
#emissions 
EM = extensions.parse(sheet_name="Emiss_act", index_col=[0,1,2], header=[0,1,2,3])
EM_FD = extensions.parse(sheet_name="Emiss_FD", index_col=[0,1,2], header=[0,1,2,3])

EM = EM.loc[Emission].sum(axis = 0)
EM_FD = EM_FD.loc[Emission]

#%%calculate the intensiteis in the baseline scenario
RE_f = RE.values @ inv_diag_x
EM_f = EM.values @ inv_diag_x
#%%Create necessary variables to calculate the new resource extractions 
I = np.eye(A_hybrid.shape[0])
L_ct = np.linalg.inv(I - A_modify.values)
Z_modify = A_modify @ (np.diag(x_hybrid))
x_ct = Z_modify.sum(axis = 1) + Y_modify.sum(axis = 1)
x_ct = L_ct @ Y_modify.values.sum(axis = 1)
RE_ct = RE_f * x_ct
EM_ct = EM_f * x_ct

F_diff_RE = (RE_ct - RE.values)#.dropna()
F_diff_EM = (EM_ct - EM.values)#.dropna()

F_diff_RE = pd.DataFrame(F_diff_RE, index = RE.index)
F_diff_RE_grouped_region = F_diff_RE.groupby(level=0, axis=0, sort=False).sum()
#F_relative_change_grouped_region *= 0.001 

F_diff_EM = pd.DataFrame(F_diff_EM, index = EM.index)
F_diff_EM_grouped_region = F_diff_EM.groupby(level=0, axis=0, sort=False).sum()

RE_FD_grouped_region = RE_FD.groupby(level=0, axis=1, sort=False).sum()
EM_FD_grouped_region = EM_FD.groupby(level=0, axis=1, sort=False).sum()


total_RE  = F_diff_RE_grouped_region.values + RE_FD_grouped_region.T.values
total_RE = pd.DataFrame(total_RE, index =F_diff_RE_grouped_region.index )

total_EM  = F_diff_EM_grouped_region.values + EM_FD_grouped_region.T.values
total_EM = pd.DataFrame(total_EM, index =F_diff_RE_grouped_region.index )


total_RE = total_RE/1000  #tonnes to kilotonnes
total_EM = total_EM/1000000 #kg to kilotonnes

#%% Make a graph that includes the below threshold values so it doesnt dissapear out of the system 
colors = plt.get_cmap('Set1').colors
ax = total_RE.unstack().plot(kind="bar", stacked=True, legend=False, figsize=(20, 12), color=colors)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 11)
ax.grid(True)  # Add grid lines
ax.set_title(f'difference baseline and ct for {resource}')
ax.set_ylabel(f"Resource extraction of {resource} in tonnes")
ax.set_xlabel('Regions')

# Show the plot
plt.show()

#%%
F_diff_RE_grouped_region.plot(kind = "bar")
#%%
threshold = 1000
F_relative_change1 = F_diff_RE.droplevel([2,3], axis=0) 

# Filter the DataFrame to include only values above the threshold
filtered_df = F_relative_change1[np.absolute(F_relative_change1) > threshold].dropna()

# Calculate the sum of values below the threshold
below_threshold_sum = F_relative_change1[np.absolute(F_relative_change1) <= threshold].sum().sum()

# Add the below-threshold sum as a new row
filtered_df.loc[('Below Threshold', 'Sum of below threshold'), :] = below_threshold_sum

# Sort the DataFrame to keep the new row at the end (optional)
# filtered_df = filtered_df.sort_index()

# Choose a color palette (using Set1)
colors = plt.get_cmap('Set1').colors
plt.rcParams.update({'font.size': 18})  # Reducing font size

# Plot the filtered DataFrame with adjusted size and legend placement
ax = filtered_df.unstack().plot(kind="bar", stacked=True, legend=False, figsize=(10, 6), color=colors)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 11)
ax.grid(True)  # Add grid lines
ax.set_title(f'Filtered difference in resource extraction (full- baseline)\n in {resource} (threshold = {threshold} {unit})', fontsize=12)
ax.set_ylabel(f"{resource} in tonnes", fontsize=12)
ax.set_xlabel('Regions',fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
# ax.set_yticklabels(ax.get_yticklabels(),fontsize=12)

# Show the plot
plt.show()

#%%
threshold = 500000
F_relative_change1 = F_diff_EM.droplevel([2,3], axis=0) 

# Filter the DataFrame to include only values above the threshold
filtered_df = F_relative_change1[np.absolute(F_relative_change1) > threshold].dropna()

# Calculate the sum of values below the threshold
below_threshold_sum = F_relative_change1[np.absolute(F_relative_change1) <= threshold].sum().sum()

# Add the below-threshold sum as a new row
filtered_df.loc[('Below Threshold', 'Sum of below threshold'), :] = below_threshold_sum

# Sort the DataFrame to keep the new row at the end (optional)
# filtered_df = filtered_df.sort_index()

# Choose a color palette (using Set1)
colors = plt.get_cmap('Set1').colors
plt.rcParams.update({'font.size': 18})  # Reducing font size

# Plot the filtered DataFrame with adjusted size and legend placement
ax = filtered_df.unstack().plot(kind="bar", stacked=True, legend=False, figsize=(10, 6), color=colors)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize = 11)
ax.grid(True)  # Add grid lines
ax.set_title(f'Filtered difference emisison of interest (full- baseline)\n in {Emission} (threshold = {threshold} {unit})', fontsize=12)
ax.set_ylabel(f"{Emission} in tonnes", fontsize=12)
ax.set_xlabel('Regions',fontsize=12)
plt.tight_layout(pad=3.0)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
# ax.set_yticklabels(ax.get_yticklabels(),fontsize=12)

# Show the plot
plt.show()

#%%
F_diff_RE_total= F_diff_RE.sum()
