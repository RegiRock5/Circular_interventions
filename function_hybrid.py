# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:53:04 2024

@author: regin
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
current_pop = 16655799
project_pop = 20610000
Populationgrowth = project_pop / current_pop
unit = "kilotonne"
#%% input path of the IOT 
iot_path = r"C:/Industrial_ecology/Thesis/IOT_2015_ixi"
save_path = r'C:/Industrial_ecology/Thesis/Circularinterventions/Code'
#%%Input of the HIOT 
# data input and modify
hiot_path = "C:/Industrial_ecology/Thesis/Circularinterventions/Data/"
Z_hybrid = pd.read_csv(f"{hiot_path}MR_HIOT_2011_v3_3_18_by_product_technology.csv", index_col=[0,1,2,3,4], header=[0,1,2,3])
Z_hybrid = pd.DataFrame(Z_hybrid.values, columns = Z_hybrid.columns, index = Z_hybrid.columns)
Z_hybrid = Z_hybrid.droplevel([2,3], axis=1).droplevel([2,3], axis=0) 
Y_hybrid = pd.read_csv(f"{hiot_path}MR_HIOT_2011_v3_3_18_FD.csv", index_col=[0,1,2,3,4], header=[0,1,2,3])
Y_hybrid = pd.DataFrame(Y_hybrid.values, columns = Y_hybrid.columns, index = Z_hybrid.columns)
Y_hybrid = Y_hybrid.droplevel([2,3], axis=1) 
#x_hybrid = Z_hybrid.sum(axis = 1) + Y_hybrid.sum(axis = 1)
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
Y_modify = Y_hybrid.copy()
#%% import extensions and slice the data you want to use
extensions = pd.ExcelFile(f"{hiot_path}MR_HIOT_2011_v3_3_18_extensions.xlsx")
extensions.sheet_names

resource = "Iron ores"
#resource = "Bauxite and aluminium ores"
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

RE_f = RE.values @ inv_diag_x
EM_f = EM.values @ inv_diag_x

tester = np.diag(RE_f) @ L_hybrid @ Y_hybrid.sum(axis =1)
#%%
file_path = 'C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/shocks_full.xlsx'
sheet_name = 'z'  # Replace with the name of your sheet
Full_shocks_A = pd.read_excel(file_path, sheet_name=sheet_name)
print(Full_shocks_A)


#%%
def apply_shocks_A_multiple_sequencesreg(file_path, A_matrix, Y_matrix, sequencesA, sequencesY, indicatorimpact, indicatorintensity, threshold,sensitivity):
    # Read the Excel file
    Full_shocks_A = pd.read_excel(file_path, sheet_name='z')
    Full_shocks_Y = pd.read_excel(file_path, sheet_name='Y')

    # Dictionary to store diffcheckers for each sequence
    diffcheckers = {}
    diffcheckerimpact = {}

    # Ensure sequenceA and sequenceY are the same length
    max_length = max(len(sequencesA), len(sequencesY))
    sequencesA.extend([[]] * (max_length - len(sequencesA)))
    sequencesY.extend([[]] * (max_length - len(sequencesY)))

    # Process sequences together
    for seq_index, (seqA, seqY) in enumerate(zip(sequencesA, sequencesY)):
        # Create copies of the original DataFrames to modify
        A_modify = A_matrix.copy()
        Y_modify = Y_matrix.copy()

        # Process sequenceA
        for seq in seqA:
            if seq < len(Full_shocks_A):
                row = Full_shocks_A.iloc[seq]
                country_row = row['row region']
                sector_row = row['row sector']
                country_column = row['column region']
                sector_column = row['column sector']
                value = row['value']
                typechange = row["type"]
                print(seq)
                if typechange == "Percentage":
                    A_modify.loc[(country_row, sector_row), (country_column, sector_column)] *= ((1 + value)*sensitivity)
                else:
                    A_modify.loc[(country_row, sector_row), (country_column, sector_column)] += (value*sensitivity)
            else:
                print(f"Index {seq} is out of range for the DataFrame.")

        # Process sequenceY
        for seq in seqY:
            if seq < len(Full_shocks_Y):
                row = Full_shocks_Y.iloc[seq]
                country_row = row['row region']
                sector_row = row['row sector']
                country_column = row['column region']
                demand_column = row['demand category']
                value = row['value']
                typechange = row["type"]
                print(seq)
                if typechange == "Percentage":
                    Y_modify.loc[(country_row, sector_row), (country_column, demand_column)] *= ((1 + value)*sensitivity)
                else:
                    Y_modify.loc[(country_row, sector_row), (country_column, demand_column)] += (value*sensitivity)
            else:
                print(f"Index {seq} is out of range for the DataFrame.")

        # Groupby to check results
        I = np.eye(A_matrix.shape[0])
        x_modify = np.linalg.inv(I - A_modify) @ Y_modify.sum(axis=1)
        x_baseline = np.linalg.inv(I - A_matrix) @ Y_matrix.sum(axis=1)

        diffchecker = pd.DataFrame()
        diffchecker["baseline"] = x_baseline
        diffchecker["changes"] = x_modify
        diffchecker["diff"] = diffchecker["changes"] - diffchecker["baseline"]
        diffcheckerdif = diffchecker["diff"]

        # Store the diffchecker in the dictionary with the sequence index as the key
        diffcheckers[f'intervention_A_{seq_index}_Y_{seq_index}'] = diffcheckerdif
        diffcheckerfull = pd.DataFrame.from_dict(diffcheckers)

        L_ct = np.linalg.inv(I - A_modify.values)
        x_ct = L_ct @ Y_modify.values.sum(axis=1)
        RE_ct = indicatorintensity * x_ct
        F_diff_RE = (RE_ct - indicatorimpact)
        diffcheckerimpact[f'intervention{seq_index}'] = F_diff_RE
        F_diff_RE = pd.DataFrame(F_diff_RE, index=A_matrix.index)
        F_relative_change1 = F_diff_RE/1000
        

        # Filter the DataFrame to include only values above the threshold
        filtered_df = F_relative_change1[np.abs(F_relative_change1) > threshold].dropna()

        # Calculate the sum of values below the threshold
        below_threshold_sum = F_relative_change1[np.abs(F_relative_change1) <= threshold].sum().sum()

        # Add the below-threshold sum as a new row
        filtered_df.loc[('Below Threshold', 'Sum of below threshold'), :] = below_threshold_sum

        # Choose a color palette (using Set1)
        colors = plt.get_cmap('Set1').colors
        plt.rcParams.update({'font.size': 18})

        # Plot the filtered DataFrame with adjusted size and legend placement
        ax = filtered_df.unstack().plot(kind="bar", stacked=True, legend=False, figsize=(10, 6), color=colors)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True)
        ax.set_title(f'Filtered differences for scenario {seq_index}\n in {indicator} (threshold = {threshold})', fontsize=12)
        ax.set_ylabel(f"{indicator} in kt", fontsize=12)
        ax.set_xlabel('Regions')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)

        # Show the plot
        plt.show()

    return diffcheckerfull, diffcheckerimpact
#%%
# Example usage
file_path = 'C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/shocks_full.xlsx'

sequencesA =[ 
    list(range(0, 31)),  # Sequence 1
    list(range(32, 53)), # Sequence 2
    list(range(0, 53)) # Sequence 3
    ]

sequencesY =[ 
    list(range(0, 2)),  # Sequence 1
    list(range(2,3)), # Sequence 2
    list(range(0, 3))# Sequence 3
    ]

indicator = 'Aluminium'
indicatorimpact = RE.values
indicatorintensity = RE_f
threshold = 2
A_hybrid1 = A_hybrid.copy()  # Replace this with your actual DataFrame

sensitivity = 1
diffcheckers = apply_shocks_A_multiple_sequencesreg(file_path, A_hybrid, Y_hybrid, sequencesA, sequencesY, indicatorimpact, indicatorintensity ,threshold,sensitivity)

sensitivity = 1 
diffcheckers,diffcheckerimpact = apply_shocks_A_multiple_sequencesreg(file_path, A_hybrid, Y_hybrid, sequencesA, sequencesY, indicatorimpact, indicatorintensity ,threshold,sensitivity)
sensitivity = 1.05
diffcheckersmin, diffcheckerminimpact  = apply_shocks_A_multiple_sequencesreg(file_path, A_hybrid, Y_hybrid, sequencesA, sequencesY, indicatorimpact, indicatorintensity ,threshold,sensitivity)
sensitivity = 1.02
diffcheckersplus,diffcheckerplusimpact = apply_shocks_A_multiple_sequencesreg(file_path, A_hybrid, Y_hybrid, sequencesA, sequencesY, indicatorimpact, indicatorintensity ,threshold,sensitivity)

#%%

diffcheckerimpact = pd.DataFrame.from_dict(diffcheckerimpact)
diffcheckerimpact.index = A_hybrid.index

diffcheckerminimpact = pd.DataFrame.from_dict(diffcheckerminimpact)
diffcheckerminimpact.index = A_hybrid.index

diffcheckerplusimpact = pd.DataFrame.from_dict(diffcheckerplusimpact)
diffcheckerplusimpact.index = A_hybrid.index


differencesensitivity= diffcheckerplusimpact-diffcheckerimpact
differencesensitivity1= diffcheckerminimpact-diffcheckerimpact

differencesensitivity.plot()
difplot = pd.DataFrame()
difplot["increase_5%"] = differencesensitivity.intervention2.values
difplot["increase_2%"] = differencesensitivity1.intervention2.values

#%%
#indicator = resource
threshold = 1
unit = "kt"
F_diff_RE = differencesensitivity.intervention2.values

F_diff_RE = pd.DataFrame(F_diff_RE, index=A_hybrid.index)
F_relative_change1 = F_diff_RE/1000

# Filter the DataFrame to include only values above the threshold
filtered_df = F_relative_change1[np.abs(F_relative_change1) > threshold].dropna()

# Calculate the sum of values below the threshold
below_threshold_sum = F_relative_change1[np.abs(F_relative_change1) <= threshold].sum().sum()

# Add the below-threshold sum as a new row
filtered_df.loc[('Below Threshold', 'Sum of below threshold'), :] = below_threshold_sum

# Choose a color palette (using Set1)
colors = plt.get_cmap('Set1').colors
plt.rcParams.update({'font.size': 18})

# Plot the filtered DataFrame with adjusted size and legend placement
ax = filtered_df.unstack().plot(kind="bar", stacked=True, legend=False, figsize=(10, 6), color=colors)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
ax.grid(True)
ax.set_title(f'Filtered differences sensitivity\n {indicator} (threshold = {threshold})', fontsize=12)
ax.set_ylabel(f"{indicator}\n ({unit})", fontsize=11)
ax.set_xlabel('Regions')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)

# Show the plot
plt.show()

#%%
diffcheckerimpact2 = diffcheckerimpact/1000
diffcheckerimpact2.groupby(level=0, axis=0, sort=False).sum()
difplot = difplot/1000
labelresource = resource
labelresource = labelresource.replace(" ", "_")
outputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/output_visuals/"
diffcheckerimpact2.to_csv(f'{outputpath}{labelresource}_hyb_impact.csv', index=True)  
difplot.to_csv(f'{outputpath}{labelresource}_hyb_sens.csv', index=True)

# #%%
# diffcheckerimpact2 = diffcheckerimpact/1000
# diffcheckerimpact2.groupby(level=0, axis=0, sort=False).sum()
# difplot = difplot/1000
# labelresource = Emission
# labelresource = labelresource[:14]
# labelresource = labelresource.replace(" ", "_")
# outputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/output_visuals/"
# diffcheckerimpact2.to_csv(f'{outputpath}{labelresource}_hyb_impact.csv', index=True)  
# difplot.to_csv(f'{outputpath}{labelresource}_hyb_sens.csv', index=True)