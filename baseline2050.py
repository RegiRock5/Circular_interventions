# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 09:55:10 2024

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
Y = pd.read_csv(f'{hiot_path}Y.txt' , sep='\t', index_col=[0, 1], header=[0, 1])
Y_org = Y.copy()
x_org = Z.sum(axis = 1) + Y.sum(axis = 1)
x_out = x_org.copy()
x_out[x_out!=0] = 1/x_out[x_out!=0]
inv_diag_x = np.diag(x_out)
A_org = Z @ inv_diag_x
I = np.identity(A_org.shape[0])
L_org = np.linalg.inv(I - A_org)
#%% import extensions and slice the data you want to use
F_sat = pd.read_csv(f'{hiot_path}satellite/F.txt' , sep='\t', index_col=[0], header=[0, 1])
indicator = "Domestic Extraction Used - Metal Ores - Bauxite and aluminium ores"
indicator ="Domestic Extraction Used - Metal Ores - Iron ores"
F_indicator = F_sat.loc[indicator]


# indicator = "Carbon dioxide (CO2) Fuel combustion"
# F_imp = pd.read_csv(f'{hiot_path}impacts/F.txt' , sep='\t', index_col=[0], header=[0, 1])
# F_indicator = F_imp.loc[indicator]

# unit = "Kt"
# indicator ="CO2 - combustion - air"
# unit = "kg"

#%%
f_indicator = F_indicator @ inv_diag_x
Y_reg = Y_org.groupby(level=0, axis=1, sort=False).sum()
CBA_e_org = np.diag(f_indicator) @ L_org @ Y_reg
CBA_e_org_country = CBA_e_org.sum(0)
CBA_e_org.index = A_org.index
#%%
Y.NL = Y.NL * Populationgrowth
x = Z.sum(axis = 1) + Y.sum(axis = 1)
x_out = x.copy()
x_out[x_out!=0] = 1/x_out[x_out!=0]
inv_diag_x = np.diag(x_out)
A = Z @ inv_diag_x
A = pd.DataFrame(A.values, columns = Z.columns, index = Z.columns)
I = np.eye(A.shape[0])
L = np.linalg.inv(I-A)
Y_modify = Y.copy()
x_pop = L @ Y_modify.sum(1)
#%%
Y_reg = Y.groupby(level=0, axis=1, sort=False).sum()
CBA_e = np.diag(f_indicator) @ L @ Y_reg
CBA_e_country = CBA_e.sum(0)
CBA_e.index = A_org.index

#%%
difference_e_pop = CBA_e - CBA_e_org

#%%
resultsdf = pd.DataFrame()

#%% FUll output of data 
resultsdf["popgrowth"] = CBA_e_country
resultsdf["orgimpact"] = CBA_e_org_country
resultsdf["difference_pop"] =  resultsdf["popgrowth"] - resultsdf["orgimpact"]

#%% plot the data 2

# # Create a figure and subplots
# fig, axs = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(30, 22))
# plt.rcParams.update({'font.size': 20})  # Reducing font size

# # Plot the data on each subplot
# resultsdf.org_output.groupby(level=0, axis=0, sort=False).sum().plot(kind='bar',ax=axs[0], label="Base gross output", color="#2F94A8")
# resultsdf.popgrowth.groupby(level=0, axis=0, sort=False).sum().plot(kind='bar',ax=axs[0], label="Population growth", color="#AD1556")
# resultsdf.difference_pop.groupby(level=0, axis=0, sort=False).sum().plot(kind='bar',ax=axs[1], label="difference population growth", color="#512B84")
# # Add legend to each subplot
# axs[0].legend()
# axs[1].legend()
# axs[1].set_xticks(range(len(region_labels)))
# axs[1].set_xticklabels(region_labels, rotation=90)

# for ax in axs:
#     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     ax.set_ylabel("In million euro's")
#     ax.set_xlabel('Regions')
#     ax.tick_params(axis='x', rotation=0)
#     ax.xaxis.set_tick_params(which='both')
#     ax.grid(True)  # Add grid lines
# # Adjust layout to prevent overlapping
# plt.tight_layout(pad=3.0)
# plt.xticks(sector_labels)

# # Show the plot
# plt.show()



#%%
# indicator = "resource"
threshold = 10
unit = "kt"
unit = "Mt"
difference_e_pop_sum = difference_e_pop.sum(1)
F_diff_RE = difference_e_pop_sum.values
CBAE_sum = CBA_e_org.sum(1)
F_diff_RE = CBAE_sum.values

F_diff_RE = pd.DataFrame(F_diff_RE, index=A.index)
F_relative_change1 = F_diff_RE/1000

# Filter the DataFrame to include only values above the threshold
filtered_df = F_relative_change1[np.abs(F_relative_change1) > threshold].dropna()

# Calculate the sum of values below the threshold
below_threshold_sum = F_relative_change1[np.abs(F_relative_change1) <= threshold].sum().sum()

# Add the below-threshold sum as a new row
filtered_df.loc[('Below Threshold', 'Sum of below threshold'), :] = below_threshold_sum
filtered_df = filtered_df.sort_values(by= [0])
# Choose a color palette (using Set1)
colors = plt.get_cmap('Set1').colors
colors = ['#E56134', '#B0B0B0'] #iron
colors = ['#47A690', '#B0B0B0'] # aluminium
plt.rcParams.update({'font.size': 25})

# Plot the filtered DataFrame with adjusted size and legend placement
ax = filtered_df.unstack().plot(kind="bar", stacked=True, legend=False, figsize=(25, 16), color=colors)
#plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
ax.set_ylim(0, filtered_df.max().max() * 1.2)


# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
ax.grid(True)
#ax.set_title(f'Filtered differences due to population growth\n {indicator} (threshold = {threshold} {unit})')
ax.set_ylabel(f"{indicator}\n ({unit})")
ax.set_xlabel('Regions')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

for p in ax.patches:
    height = p.get_height()
    if height != 0:
        ax.annotate(f'{height:.1f}{unit}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', fontsize=24, rotation = 90, xytext=(0, 70), textcoords='offset points')

# Show the plot
plt.show()


print(filtered_df.sum())

#%%
# threshold = 0
# unit = "kt"
# unit = "Mt"
# difference_e_pop_sum = difference_e_pop.sum(0)
# F_diff_RE = difference_e_pop_sum#.values
# CBAE_sum = CBA_e.sum(0)
# F_diff_RE = CBAE_sum#.values

# # F_diff_RE = pd.DataFrame(F_diff_RE, index=A.index)
# F_relative_change1 = F_diff_RE/1000

# # Filter the DataFrame to include only values above the threshold
# filtered_df = F_relative_change1[np.abs(F_relative_change1) > threshold].dropna()

# # Calculate the sum of values below the threshold
# below_threshold_sum = F_relative_change1[np.abs(F_relative_change1) <= threshold].sum().sum()

# # Add the below-threshold sum as a new row
# # filtered_df.loc[('Below Threshold', 'Sum of below threshold',"nothing", "nothing"), :] = below_threshold_sum
# filtered_df.loc['Below Threshold'] = below_threshold_sum

# filtered_df = filtered_df.sort_values()#(by= [0])

# filtered_df1 = filtered_df#.droplevel([2,3], axis=0) 

# # Choose a color palette (using Set1)
# # colors = plt.get_cmap('Set1').colors
# colors = ['#E56134', '#B0B0B0'] #iron
# #colors = ['#47A690', '#B0B0B0'] # aluminium
# plt.rcParams.update({'font.size': 25})

# # Plot the filtered DataFrame with adjusted size and legend placement
# # ax = filtered_df1.unstack().plot(kind="bar", stacked=True, legend=False, figsize=(25, 16), color=colors)
# ax = filtered_df1.plot(kind="bar", stacked=True, legend=False, figsize=(25, 16), color=colors)

# #plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
# ax.set_ylim(0, filtered_df1.max().max() * 1.2)


# # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
# ax.grid(True)
# #ax.set_title(f'Filtered differences due to population growth\n {indicator} (threshold = {threshold} {unit})')
# ax.set_ylabel(f"{indicator}\n ({unit})")
# ax.set_xlabel('Regions')
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# for p in ax.patches:
#     height = p.get_height()
#     if height != 0:
#         ax.annotate(f'{height:.1f}{unit}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', fontsize=24, rotation = 90, xytext=(0, 70), textcoords='offset points')

# # Show the plot
# plt.show()


# print(filtered_df.sum())


#%%

# resultsdf = resultsdf/1000
# labelresource = "Bauxite and aluminium ores"
# labelresource = labelresource.replace(" ", "_")
# outputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/output_visuals/projection/"
# resultsdf.to_csv(f'{outputpath}{labelresource}_mon_2050.csv', index=True)  

#%%
# resultsdf = resultsdf/1000
# labelresource = "CO2"
# # labelresource = labelresource[:14]
# labelresource = labelresource.replace(" ", "_")
# outputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/output_visuals/projection/"
# resultsdf.to_csv(f'{outputpath}{labelresource}_mon_2050.csv', index=True)  


#%% new 
labelresource = "Iron ores"
labelresource = labelresource.replace(" ", "_")
outputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/output_visuals/projection/version2/"
filtered_df.to_csv(f'{outputpath}{labelresource}_mon_2050.csv', index=True) 