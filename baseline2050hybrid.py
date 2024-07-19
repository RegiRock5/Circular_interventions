# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:04:32 2024

@author: regin
"""# -*- coding: utf-8 -*-
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
#%%Input of the HIOT 
# data input and modify
hiot_path = "C:/Industrial_ecology/Thesis/Circularinterventions/Data/"
Z = pd.read_csv(f"{hiot_path}MR_HIOT_2011_v3_3_18_by_product_technology.csv", index_col=[0,1,2,3,4], header=[0,1,2,3])
Z_hybrid = pd.DataFrame(Z.values, columns = Z.columns, index = Z.columns)
Z_hybrid = Z_hybrid.droplevel([2,3], axis=1).droplevel([2,3], axis=0) 
Y = pd.read_csv(f"{hiot_path}MR_HIOT_2011_v3_3_18_FD.csv", index_col=[0,1,2,3,4], header=[0,1,2,3])
# Y_hybrid = pd.DataFrame(Y_hybrid.values, columns = Y_hybrid.columns, index = Z_hybrid.columns)
# Y_hybrid = Y_hybrid.droplevel([2,3], axis=1) 
#x_hybrid = Z_hybrid.sum(axis = 1) + Y_hybrid.sum(axis = 1)

#%%

Y_org = Y.copy()
x_org = Z.sum(axis = 1) + Y.sum(axis = 1)
x_out = x_org.copy()
x_out[x_out!=0] = 1/x_out[x_out!=0]
inv_diag_x = np.diag(x_out)
A_org = Z @ inv_diag_x
I = np.identity(A_org.shape[0])
L_org = np.linalg.inv(I - A_org)


#%%
extensions = pd.ExcelFile(f"{hiot_path}MR_HIOT_2011_v3_3_18_extensions.xlsx")
extensions.sheet_names

resource = "Iron ores"
resource = "Bauxite and aluminium ores"
#resource = "Copper ores"
indicator = resource
#resource extraction --> take only the material of interest
RE = extensions.parse(sheet_name="resource_act", index_col=[0,1], header=[0,1,2,3]) 
RE_FD = extensions.parse(sheet_name="resource_FD", index_col=[0,1], header=[0,1,2,3]) 
RE = RE.loc[resource].sum(axis = 0)
RE_FD = RE_FD.loc[resource]
#emissions 
RE_f = RE.values @ inv_diag_x


# Emission = "Carbon dioxide, fossil"
# #resource = "Copper ores"
# indicator = Emission
# #resource extraction --> take only the material of interest
# #emissions 
# EM = extensions.parse(sheet_name="Emiss_act", index_col=[0,1,2], header=[0,1,2,3])
# EM_FD = extensions.parse(sheet_name="Emiss_FD", index_col=[0,1,2], header=[0,1,2,3])

# EM = EM.loc[Emission].sum(axis = 0)
# EM_FD = EM_FD.loc[Emission]

# RE_f = EM.values @ inv_diag_x


#%%
# F_indicator = F_sat.loc[indicator]
# f_indicator = F_indicator @ inv_diag_x

f_indicator = RE_f.copy()
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
difference_e_pop.groupby(level=0, axis=1, sort=False).sum().sum(axis =0)
difference_e_pop.sum(0)
#%%
resultsdf = pd.DataFrame()

#%% FUll output of data 
resultsdf["popgrowth"] = CBA_e_country
resultsdf["orgimpact"] = CBA_e_org_country
resultsdf["difference_pop"] =  resultsdf["popgrowth"] - resultsdf["orgimpact"]

#%% plot the data 2
# threshold = 0
# unit = "kt"
# unit = "Mt"
# difference_e_pop_sum = difference_e_pop.sum(0)
# F_diff_RE = difference_e_pop_sum#.values
# CBAE_sum = CBA_e.sum(0)
# F_diff_RE = CBAE_sum#.values

# # F_diff_RE = pd.DataFrame(F_diff_RE, index=A.index)
# F_relative_change1 = F_diff_RE/1000000

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


# #%%

# # resultsdf = resultsdf/1000000
# # labelresource = resource
# # labelresource = labelresource.replace(" ", "_")
# # outputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/output_visuals/projection/"
# # resultsdf.to_csv(f'{outputpath}{labelresource}_hyb_2050.csv', index=True)  

# #%%
# resultsdf = resultsdf/1000000
# labelresource = "CO2"
# labelresource = labelresource[:14]
# labelresource = labelresource.replace(" ", "_")
# outputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/output_visuals/projection/"
# resultsdf.to_csv(f'{outputpath}{labelresource}_hyb_2050.csv', index=True)  


#%%
threshold = 5
unit = "kt"
unit = "Mt"
difference_e_pop_sum = difference_e_pop.sum(1)
F_diff_RE = difference_e_pop_sum.values
CBAE_sum = CBA_e_org.sum(1)
F_diff_RE = CBAE_sum.values

F_diff_RE = pd.DataFrame(F_diff_RE, index=A.index)
F_relative_change1 = F_diff_RE/1000000

# Filter the DataFrame to include only values above the threshold
filtered_df = F_relative_change1[np.abs(F_relative_change1) > threshold].dropna()

# Calculate the sum of values below the threshold
below_threshold_sum = F_relative_change1[np.abs(F_relative_change1) <= threshold].sum().sum()
filtered_df1 = filtered_df.droplevel([1,2,3], axis=0) 
filtered_df1

# Add the below-threshold sum as a new row
# filtered_df.loc[('Below Threshold', 'Sum of below threshold'), :] = below_threshold_sum
filtered_df1.loc['Below Threshold'] = below_threshold_sum

filtered_df1 = filtered_df1.sort_values(by= [0])
# Choose a color palette (using Set1)
colors = plt.get_cmap('Set1').colors
colors = ['#E56134', '#B0B0B0'] #iron
colors = ['#47A690', '#B0B0B0'] # aluminium
plt.rcParams.update({'font.size': 25})

# Plot the filtered DataFrame with adjusted size and legend placement
ax = filtered_df1.unstack().plot(kind="bar", stacked=True, legend=False, figsize=(25, 16), color=colors)
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
labelresource = "Bauxite and aluminium ores"#"Iron ores"
labelresource = labelresource.replace(" ", "_")
outputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/output_visuals/projection/version2/"
filtered_df1.to_csv(f'{outputpath}{labelresource}_hyb_2050.csv', index=True) 