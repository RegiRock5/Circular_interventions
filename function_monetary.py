# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 16:56:27 2024

@author: regin
"""

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
#%%Input of the HIOT 
# data input and modify
hiot_path = "C:/Industrial_ecology/Thesis/IOT_2011_ixi/"
Z = pd.read_csv(f'{hiot_path}Z.txt', sep='\t', index_col=[0, 1], header=[0, 1])
#Z = pd.DataFrame(Z.values, columns = Z.columns, index = Z.columns)
# A = pd.read_csv(f'{path}A.txt', sep='\t', index_col=[0, 1], header=[0, 1])
#Z = Z.droplevel([2,3], axis=1).droplevel([2,3], axis=0) 
Y = pd.read_csv(f'{hiot_path}Y.txt' , sep='\t', index_col=[0, 1], header=[0, 1])
#Y = pd.DataFrame(Y.values, columns = Y.columns, index = Z.columns)
#x = Z.sum(axis = 1) + Y.sum(axis = 1)
#Y = Y.droplevel([2,3], axis=1) 
Y.NL = Y.NL * Populationgrowth

x = Z.sum(axis = 1) + Y.sum(axis = 1)
#modify the final demand to project to 2050
x_out = x.copy()
x_out[x_out!=0] = 1/x_out[x_out!=0]
inv_diag_x = np.diag(x_out)
A = Z @ inv_diag_x
A = pd.DataFrame(A.values, columns = Z.columns, index = Z.columns)
I = np.eye(A.shape[0])

L = np.linalg.inv(I-A)
Y_modify = Y.copy()

x_pop = L@ Y_modify.sum(1)
#%%

labels = Y.index.get_level_values(1).tolist()

def insert_break_after_40(text):
    """Inserts a line break after the first space encountered beyond the 40th character."""
    if len(text) <= 40:
        return text
    space_index = text.find(' ', 40)
    if space_index == -1:
        return text  # No space found after the 40th character, return as is.
    return text[:space_index] + '\n' + text[space_index+1:]

# Modify the labels list
modified_labels = [insert_break_after_40(label) for label in labels]



#%% import extensions and slice the data you want to use
F_sat = pd.read_csv(f'{hiot_path}satellite/F.txt' , sep='\t', index_col=[0], header=[0, 1])
F_imp = pd.read_csv(f'{hiot_path}impacts/F.txt' , sep='\t', index_col=[0], header=[0, 1])

indicator = "Domestic Extraction Used - Metal Ores - Bauxite and aluminium ores"
indicator ="Domestic Extraction Used - Metal Ores - Iron ores"
indicator = "GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)"
unit = "kg"
# indicator ="CO2 - combustion - air"
# unit = "kg"


# F_indicator = F_sat.loc[indicator]
# f_indicator = F_indicator @ inv_diag_x

F_indicator = F_imp.loc[indicator]
f_indicator = F_indicator @ inv_diag_x

# F_indicator = pd.DataFrame(f_indicator *  x_pop, index = F_sat.columns) # calcualte impact adjusted to pupulation growht
#%%
# file_path = 'C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/shocks_full.xlsx'
# sheet_name = 'z'  # Replace with the name of your sheet
# Full_shocks_A = pd.read_excel(file_path, sheet_name=sheet_name)
# print(Full_shocks_A)

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
                    A_modify.loc[(country_row, sector_row), (country_column, sector_column)] += (value * sensitivity)
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
                    Y_modify.loc[(country_row, sector_row), (country_column, demand_column)] += (value * sensitivity)
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
        diffcheckerimpact[f'intervention_A_{seq_index}_Y_{seq_index}'] = F_diff_RE
        #diffcheckerimpact = pd.DataFrame.from_dict(diffcheckerimpact)

        F_diff_RE = pd.DataFrame(F_diff_RE, index=A_matrix.index)
        F_relative_change1 = F_diff_RE# /1000000  # uncomment this when working with co2 as it is given in kg

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
        ax.set_title(f'Filtered differences for scenario {seq_index}\n in {indicator} (threshold = {threshold})', fontsize=12)
        ax.set_ylabel(f"{indicator} ({unit})", fontsize=11)
        ax.set_xlabel('Regions')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)

        # Show the plot
        plt.show()

    return diffcheckerfull,diffcheckerimpact


#%%
# Example usage
file_path = 'C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/shocks_full.xlsx'

sequencesA=[ 
    list(range(0, 32)),  # Sequence 1
    list(range(32, 53)),
    list(range(0, 53))# Sequence 2
    ]

sequencesY =[ 
    list(range(0, 2)),  # Sequence 1
    list(range(2,3)),
    list(range(0, 3))# Sequence 2
    ]

indicatorimpact = F_indicator.values
indicatorintensity = f_indicator
threshold = 1
sensitivity = 1 
A1 = A.copy()  # Replace this with your actual DataFrame
diffcheckers,diffcheckerimpact = apply_shocks_A_multiple_sequencesreg(file_path, A, Y, sequencesA, sequencesY, indicatorimpact, indicatorintensity ,threshold,sensitivity)
sensitivity = 1.05
diffcheckersmin, diffcheckerminimpact  = apply_shocks_A_multiple_sequencesreg(file_path, A, Y, sequencesA, sequencesY, indicatorimpact, indicatorintensity ,threshold,sensitivity)
sensitivity = 1.02
diffcheckersplus,diffcheckerplusimpact = apply_shocks_A_multiple_sequencesreg(file_path, A, Y, sequencesA, sequencesY, indicatorimpact, indicatorintensity ,threshold,sensitivity)

#%%

diffcheckerimpact = pd.DataFrame.from_dict(diffcheckerimpact)
diffcheckerimpact.index = A.index

diffcheckerminimpact = pd.DataFrame.from_dict(diffcheckerminimpact)
diffcheckerminimpact.index = A.index

diffcheckerplusimpact = pd.DataFrame.from_dict(diffcheckerplusimpact)
diffcheckerplusimpact.index = A.index

differencesensitivity= diffcheckerplusimpact-diffcheckerimpact

differencesensitivity.plot()
differencesensitivity1= diffcheckerminimpact-diffcheckerimpact

difplot = pd.DataFrame()
difplot["increase_5%"] = differencesensitivity.intervention_A_2_Y_2.values
difplot["increase_2%"] = differencesensitivity1.intervention_A_2_Y_2.values

#%%
modified_labels = [insert_break_after_40(label) for label in labels]
regions_labels = Y.index.get_level_values(0).tolist()
index2 = pd.MultiIndex.from_arrays([regions_labels, modified_labels], names=['regions', 'sectors'])

threshold = 25
unit = "kt"
F_diff_RE = difplot.values#differencesensitivity.intervention_A_2_Y_2.values


F_diff_RE = pd.DataFrame(F_diff_RE, index2)# A.index)
F_relative_change1 = F_diff_RE/1000000

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

#%% export data 
# labelresource = indicator[40:]
# labelresource = labelresource.replace(" ", "_")
# #labelresource = "CO2"
# outputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/output_visuals/"
# diffcheckerimpact.to_csv(f'{outputpath}{labelresource}_mon_impact.csv', index=True)  
# difplot.to_csv(f'{outputpath}{labelresource}_mon_sens.csv', index=True)


#%%
diffcheckerimpact = diffcheckerimpact/1e6
difplot = difplot/1e6
labelresource = "CO2"
outputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/output_visuals/"
diffcheckerimpact.to_csv(f'{outputpath}{labelresource}_mon_impact.csv', index=True)  
difplot.to_csv(f'{outputpath}{labelresource}_mon_sens.csv', index=True)