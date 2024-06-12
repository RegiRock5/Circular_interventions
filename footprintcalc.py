# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:15:16 2024

@author: regin
"""
# Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%% Import all used data (baseline and all 3 scenarios)
path = r"C:/Industrial_ecology/Thesis/IOT_2011_ixi/"
outputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/Code/Output/" 
indicator = "Domestic Extraction Used - Metal Ores - Bauxite and aluminium ores"
indicator ="Domestic Extraction Used - Metal Ores - Iron ores"
unit = "Kt"
# indicator ="CO2 - combustion - air"
#unit = "kg"
#unit is given in CO2 is given in kg and domestic extraction in kt
#%%
current_pop = 16655799	
project_pop = 20610000
Populationgrowth = project_pop / current_pop
#%%
#"C:\Industrial_ecology\Thesis\Circularinterventions\Code\Input_circular_interventions\newZ.csv"
Y = pd.read_csv(f'{path}Y.txt' , sep='\t', index_col=[0, 1], header=[0, 1])
Y_org = Y.copy()
A = pd.read_csv(f'{path}A.txt', sep='\t', index_col=[0, 1], header=[0, 1])
Y.NL = Y.NL * Populationgrowth

Y_Al = pd.read_csv(f'{outputpath}Y_Al_adjusted.csv' , sep=',',index_col=[0,1,2], header=[0,2])
A_Al = pd.read_csv(f"{outputpath}A_Al_adjusted.csv", sep=',',index_col=[0,1,2], header=[0, 2])
Y_Al.NL = Y_Al.NL * Populationgrowth

Y_St = pd.read_csv(f'{outputpath}Y_St_adjusted.csv' , sep=',',index_col=[0,1,2], header=[0,2])
A_St = pd.read_csv(f"{outputpath}A_St_adjusted.csv", sep=',',index_col=[0,1,2], header=[0, 2])
Y_St.NL = Y_St.NL * Populationgrowth

Y_full = pd.read_csv(f'{outputpath}Y_full_adjusted.csv' , sep=',',index_col=[0,1,2], header=[0,2])
A_full = pd.read_csv(f"{outputpath}A_full_adjusted.csv", sep=',',index_col=[0,1,2], header=[0, 2])
Y_full.NL = Y_full.NL * Populationgrowth

Y.index = Y.index
A.index = A.index
#%%
# Import satellite accounts
F_sat = pd.read_csv(f'{path}satellite/F.txt' , sep='\t', index_col=[0], header=[0, 1])
F_sat_hh = pd.read_csv(f'{path}satellite/F_hh.txt' , sep='\t', index_col=[0], header=[0, 1])

F_sat_Al = pd.read_csv(f'{outputpath}F_Al_adjusted.csv' , sep=',', index_col=[0], header=[0, 2])
F_sat_St = pd.read_csv(f'{outputpath}F_St_adjusted.csv' , sep=',', index_col=[0], header=[0, 2])
F_sat_full = pd.read_csv(f'{outputpath}F_full_adjusted.csv' , sep=',', index_col=[0], header=[0, 2])
#%% Import impact accounts
F_imp = pd.read_csv(f'{path}impacts/F.txt' , sep='\t', index_col=[0], header=[0, 1])
F_imp_hh = pd.read_csv(f'{path}impacts/F_hh.txt' , sep='\t', index_col=[0], header=[0, 1])
#%% Perform CBA and PBA for baseline (2021)

#CBA prepare data
I = np.identity(A.shape[0])
L = np.linalg.inv(I-A)
x = L @ Y.sum(axis=1)

x_inv = x.copy()
x_inv[x_inv!=0] = 1/x_inv[x_inv!=0]

F_indicator = F_sat.loc[indicator]
F_hh_indicator = F_sat_hh.loc[indicator]

# calculate intensities
f_indicator = F_indicator @ np.diag(x_inv)
Y_reg = Y.groupby(level=0, axis=1, sort=False).sum()

#Consumption modeling baseline
CBA_baseline = f_indicator @ L @ Y_reg + F_hh_indicator.groupby(level=0, axis=0, sort=False).sum()
CBA_baseline.sort_values().iloc[[0, -1]]

#PBA calculations 
PBA_baseline = np.diag(f_indicator) @ x
PBA_baseline = pd.DataFrame(PBA_baseline)
PBA_baseline.index = A.index
PBA_baseline = PBA_baseline.groupby(level=0, axis=0, sort=False).sum()

#%% Perform CBA and PBA for scenario 1 

#CBA prepare data
I = np.identity(A_Al.shape[0])
L = np.linalg.inv(I-A_Al)
x = L @ Y_Al.sum(axis=1)

x_inv = x.copy()
x_inv[x_inv!=0] = 1/x_inv[x_inv!=0]

F_indicator = F_sat_Al.loc[indicator]
F_hh_indicator = F_sat_hh.loc[indicator]

# calculate intensities
f_indicator = F_indicator @ np.diag(x_inv)
Y_reg = Y_Al.groupby(level=0, axis=1, sort=False).sum()

#Consumption modeling baseline
CBA_Al = f_indicator @ L @ Y_reg + F_hh_indicator.groupby(level=0, axis=0, sort=False).sum()
CBA_Al.sort_values().iloc[[0, -1]]

#PBA calculations 
PBA_Al = np.diag(f_indicator) @ x
PBA_Al = pd.DataFrame(PBA_Al)
PBA_Al.index = A.index
PBA_Al = PBA_Al.groupby(level=0, axis=0, sort=False).sum()

#%% Perform CBA and PBA for scenario 2

#CBA prepare data
I = np.identity(A_St.shape[0])
L = np.linalg.inv(I-A_St)
x = L @ Y_St.sum(axis=1)

x_inv = x.copy()
x_inv[x_inv!=0] = 1/x_inv[x_inv!=0]

F_indicator = F_sat_St.loc[indicator]
F_hh_indicator = F_sat_hh.loc[indicator]

# calculate intensities
f_indicator = F_indicator @ np.diag(x_inv)
Y_reg = Y_St.groupby(level=0, axis=1, sort=False).sum()

#Consumption modeling baseline
CBA_St = f_indicator @ L @ Y_reg + F_hh_indicator.groupby(level=0, axis=0, sort=False).sum()
CBA_St.sort_values().iloc[[0, -1]]

#PBA calculations 
PBA_St = np.diag(f_indicator) @ x
PBA_St = pd.DataFrame(PBA_St)
PBA_St.index = A.index
PBA_St = PBA_St.groupby(level=0, axis=0, sort=False).sum()

#%% Perform CBA and PBA for scenario 3

#CBA prepare data
I = np.identity(A_full.shape[0])
L = np.linalg.inv(I-A_full)
x = L @ Y_full.sum(axis=1)

x_inv = x.copy()
x_inv[x_inv!=0] = 1/x_inv[x_inv!=0]

F_indicator = F_sat_full.loc[indicator]
F_hh_indicator = F_sat_hh.loc[indicator]

# calculate intensities
f_indicator = F_indicator @ np.diag(x_inv)
Y_reg = Y_full.groupby(level=0, axis=1, sort=False).sum()

#Consumption modeling baseline
CBA_full = f_indicator @ L @ Y_reg + F_hh_indicator.groupby(level=0, axis=0, sort=False).sum()
CBA_full.sort_values().iloc[[0, -1]]

#PBA calculations 
PBA_full = np.diag(f_indicator) @ x
PBA_full = pd.DataFrame(PBA_full)
PBA_full.index = A.index
PBA_full = PBA_full.groupby(level=0, axis=0, sort=False).sum()


#%% Create one dataframe containing all country emmissions
CBAframe = {"baseline": CBA_baseline,
         "scenario_Al": CBA_Al,
         "scenario_St": CBA_St,
         "scenario_both": CBA_full}

CBAdf = pd.DataFrame(CBAframe)
CBAdf["differ"] = CBAdf["scenario_both"] - CBAdf["baseline"]
CBAdf["difference"] = CBAdf["scenario_St"] - CBAdf["baseline"]
CBAdf["difference2"] = CBAdf["scenario_Al"] - CBAdf["baseline"]

#CBAdf.plot()
PBA_baseline = PBA_baseline.squeeze()
PBA_Al = PBA_Al.squeeze()
PBA_St = PBA_St.squeeze()
PBA_full = PBA_full.squeeze()


PBAframe = {"baseline": PBA_baseline,
         "scenario_Al": PBA_Al,
         "scenario_St": PBA_St,
         "scenario_both": PBA_full}

PBAdf = pd.DataFrame(PBAframe)
PBAdf["differ"] = PBAdf["scenario_both"] - PBAdf["baseline"]
PBAdf["difference"] = PBAdf["scenario_St"] - PBAdf["baseline"]
PBAdf["difference2"] = PBAdf["scenario_Al"] - PBAdf["baseline"]

#PBAdf.baseline.plot("bar")

#%%Plotting results CBA
fig, axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(25, 22))
plt.rcParams.update({'font.size': 8})  # Reducing font size

# Plot the data on each subplot
CBAdf.baseline.plot(ax=axs[0], label="CBA baseline", color = "#2F94A8")
CBAdf.scenario_Al.plot(ax=axs[1], label="CBA Al", color="#AD1556")
CBAdf.scenario_St.plot(ax=axs[2], label="CBA St", color="#000E75")
CBAdf.scenario_both.plot(ax=axs[3], label="CBA both", color="#512B84")
PBAdf.baseline.plot(ax=axs[0], label="PBA baseline", color = "#BA6418")
PBAdf.scenario_Al.plot(ax=axs[1], label="PBA Al", color="#126217")
PBAdf.scenario_St.plot(ax=axs[2], label="PBA St", color="#29DCA4")
PBAdf.scenario_both.plot(ax=axs[3], label="PBA both", color="#FFE809")

# CBAdf.differ.plot(ax=axs[0], label="CBA baseline", color = "darkorange")
# PBAdf.differ.plot(ax=axs[1], label="CBA Al", color="red")

# axs[0].bar(CBAdf.index, CBAdf['differ'], label="Difference in CBA", color="darkorange")
# axs[1].bar(PBAdf.index, PBAdf['differ'], label="difference in PBA", color="blue")
# axs[0].bar(CBAdf.index, CBAdf['difference'], label="Difference in CBA st", color="red")
# axs[1].bar(PBAdf.index, PBAdf['difference'], label="difference in PBA st", color="green")
# axs[0].bar(CBAdf.index, CBAdf['difference2'], label="Difference in CBA AL", color="gold")
# axs[1].bar(PBAdf.index, PBAdf['difference2'], label="difference in PBA AL", color="magenta")
# Add legend to each subplot
axs[0].legend()
axs[1].legend()
axs[2].legend()
axs[3].legend()


bar_width = 0.25

axs[0].set_title('Baseline CBA and PBA')
axs[1].set_title('Aluminium scenario CBA and PBA')
axs[2].set_title('Steel scenario CBA and PBA')
axs[3].set_title('Full scenario CBA and PBA')


# Set title for the entire set of subplots
fig.suptitle(f'comparison between the different scenarios in {indicator}', fontsize=16)


# for ax in axs:
#     ax.legend()

# #tickvalues = range(0,len(sector_labels))

for ax in axs:
    ax.set_ylabel("In million euro's")
    ax.set_xlabel('Regions')
    
for ax in axs:
    ax.set_xticks([r + bar_width for r in range(len(CBAdf))])
    ax.set_xticklabels(CBAdf.index)
    ax.grid(True)  # Add grid lines

# # Adjust layout to prevent overlapping
plt.tight_layout(pad=3.0)
# #plt.xticks(range(0,len(sector_labels.index)), sector_labels.index)

# Show the plot
plt.show()

#%%

# Number of bars
n_bars = 3
bar_width = 0.25

# Create subplots
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(25, 22))
plt.rcParams.update({'font.size': 8})  # Reducing font size

# Calculate bar positions
r1 = np.arange(len(CBAdf))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Plot the data on each subplot as bar charts with bars next to each other
axs[0].bar(r1, CBAdf['differ'], width=bar_width, label="Difference in CBA", color="#2F94A8")
axs[0].bar(r2, CBAdf['difference'], width=bar_width, label="Difference in CBA St", color="#AD1556")
axs[0].bar(r3, CBAdf['difference2'], width=bar_width, label="Difference in CBA AL", color="#FFE809")

axs[1].bar(r1, PBAdf['differ'], width=bar_width, label="Difference in PBA", color="#512B84")
axs[1].bar(r2, PBAdf['difference'], width=bar_width, label="Difference in PBA St", color="#BA6418")
axs[1].bar(r3, PBAdf['difference2'], width=bar_width, label="Difference in PBA AL", color="#126217")

axs[0].set_title('CBA Differences')
axs[1].set_title('PBA Differences')

# Set title for the entire set of subplots
fig.suptitle(f'Comparison of CBA and PBA Differences in {indicator}', fontsize=16)


# Add legend to each subplot
axs[0].legend()
axs[1].legend()

# Adjust x-axis labels rotation for better readability
for ax in axs:
    ax.set_xticks([r + bar_width for r in range(len(CBAdf))])
    ax.set_xticklabels(CBAdf.index)
    ax.grid(True)  # Add grid lines
    
for ax in axs:
    ax.set_ylabel("In million euro's")
    ax.set_xlabel('Regions')
# Adjust layout to prevent overlapping
plt.tight_layout(pad=3.0)

# Show the plot
plt.show()


#%%
#%% Perform contribution analysis on sectoral level. Removegroup by 

#CBA prepare data Baseline 
I = np.identity(A.shape[0])
L = np.linalg.inv(I-A)
x = L @ Y.sum(axis=1)

x_inv = x.copy()
x_inv[x_inv!=0] = 1/x_inv[x_inv!=0]

F_indicator = F_sat.loc[indicator]
F_hh_indicator = F_sat_hh.loc[indicator]

# calculate intensities
f_indicator = F_indicator @ np.diag(x_inv)
Y_reg = Y.groupby(level=0, axis=1, sort=False).sum()

#Consumption modeling baseline
CBA_baseline = f_indicator @ L @ np.diag(Y_reg.sum(axis =1)) #+ F_hh_indicator.groupby(level=0, axis=0, sort=False).sum()
# CBA_baseline.sort_values().iloc[[0, -1]]
CBA_baseline = pd.DataFrame(CBA_baseline, index = A.index)


#PBA calculations 
PBA_baseline = np.diag(f_indicator) @ x
PBA_baseline = pd.DataFrame(PBA_baseline)
PBA_baseline.index = A.index


#%% check baseline divison
threshold = 10000  # Set your threshold value here

# Filter the DataFrame to include only values above the threshold
filtered_df = CBA_baseline[CBA_baseline > threshold].dropna()

# Plot the filtered DataFrame with adjusted size and legend placement
ax = filtered_df.unstack().plot(kind="bar", stacked=True, legend=False, figsize=(10, 6))
ax.grid(True)  # Add grid lines
ax.set_ylabel("In million euro's")
ax.set_title(f"Filtered baseline CBA results for {indicator} ")
ax.set_xlabel('Regions')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()

#%%Make the CBA and PBA to also include the Sectoral resultion for further analysis 
I = np.identity(A_full.shape[0])
L = np.linalg.inv(I-A_full)
x = L @ Y_full.sum(axis=1)

x_inv = x.copy()
x_inv[x_inv!=0] = 1/x_inv[x_inv!=0]

F_indicator = F_sat_full.loc[indicator]
F_hh_indicator = F_sat_hh.loc[indicator]

# calculate intensities
f_indicator = F_indicator @ np.diag(x_inv)
Y_reg = Y_full.groupby(level=0, axis=1, sort=False).sum()

#Consumption modeling baseline
CBA_full = f_indicator @ L @ np.diag(Y_reg.sum(axis =1)) #+ F_hh_indicator.groupby(level=0, axis=0, sort=False).sum()
#CBA_full.sort_values().iloc[[0, -1]]
CBA_full = pd.DataFrame(CBA_full, index = A.index)

#PBA calculations 
PBA_full = np.diag(f_indicator) @ x
PBA_full = pd.DataFrame(PBA_full)
PBA_full.index = A.index


diffcheckerCBA = CBA_full - CBA_baseline
diffcheckerPBA = PBA_full - PBA_baseline
#%% Make a graph that includes the below threshold values so it doesnt dissapear out of the system 
threshold = 0.1
# Filter the DataFrame to include only values above the threshold
filtered_df = diffcheckerPBA[np.absolute(diffcheckerPBA) > threshold].dropna()

# Calculate the sum of values below the threshold
below_threshold_sum = diffcheckerPBA[np.absolute(diffcheckerPBA) <= threshold].sum().sum()

# Add the below-threshold sum as a new row
filtered_df.loc[('Below Threshold', 'Sum of below threshold'), :] = below_threshold_sum

# Sort the DataFrame to keep the new row at the end (optional)
# filtered_df = filtered_df.sort_index()

# Choose a color palette (using Set1)
colors = plt.get_cmap('Set1').colors

# Plot the filtered DataFrame with adjusted size and legend placement
ax = filtered_df.unstack().plot(kind="bar", stacked=True, legend=False, figsize=(10, 6), color=colors)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(True)  # Add grid lines
ax.set_title(f'Filtered difference in PBA (combined interventions- baseline) in {indicator}')
ax.set_ylabel(f"{indicator} ({unit})")
ax.set_xlabel('Regions')

# Show the plot
plt.show()

#%% Make a graph that includes the below threshold values so it doesnt dissapear out of the system 
threshold = 0.1

# Filter the DataFrame to include only values above the threshold
filtered_df = diffcheckerCBA[np.absolute(diffcheckerCBA) > threshold].dropna()

# Calculate the sum of values below the threshold
below_threshold_sum = diffcheckerCBA[np.absolute(diffcheckerCBA) <= threshold].sum().sum()

# Add the below-threshold sum as a new row
filtered_df.loc[('Below Threshold', 'Sum of below threshold'), :] = below_threshold_sum

# Sort the DataFrame to keep the new row at the end (optional)
# filtered_df = filtered_df.sort_index()

# Choose a color palette (using Set1)
colors = plt.get_cmap('Set1').colors

# Plot the filtered DataFrame with adjusted size and legend placement
ax = filtered_df.unstack().plot(kind="bar", stacked=True, legend=False, figsize=(10, 6), color=colors)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(True)  # Add grid lines
ax.set_title(f'Filtered difference in CBA (combined interventions - baseline) in {indicator}')
ax.set_ylabel(f"{indicator} ({unit})")
ax.set_xlabel('Regions')

# Show the plot
plt.show()

#%%
#%%Make the CBA and PBA to also include the Sectoral resultion for further analysis only for scenario Al
I = np.identity(A_Al.shape[0])
L = np.linalg.inv(I-A_Al)
x = L @ Y_Al.sum(axis=1)

x_inv = x.copy()
x_inv[x_inv!=0] = 1/x_inv[x_inv!=0]

F_indicator = F_sat_Al.loc[indicator]
F_hh_indicator = F_sat_hh.loc[indicator]

# calculate intensities
f_indicator = F_indicator @ np.diag(x_inv)
Y_reg = Y_Al.groupby(level=0, axis=1, sort=False).sum()

#Consumption modeling baseline
CBA_Al = f_indicator @ L @ np.diag(Y_reg.sum(axis =1)) #+ F_hh_indicator.groupby(level=0, axis=0, sort=False).sum()
#CBA_full.sort_values().iloc[[0, -1]]
CBA_Al = pd.DataFrame(CBA_full, index = A.index)

#PBA calculations 
PBA_Al = np.diag(f_indicator) @ x
PBA_Al = pd.DataFrame(PBA_Al)
PBA_Al.index = A.index


diffcheckerCBA = CBA_Al - CBA_baseline
diffcheckerPBA = PBA_Al - PBA_baseline
#%% Make a graph that includes the below threshold values so it doesnt dissapear out of the system 
threshold = 0.5
# Filter the DataFrame to include only values above the threshold
filtered_df = diffcheckerPBA[np.absolute(diffcheckerPBA) > threshold].dropna()

# Calculate the sum of values below the threshold
below_threshold_sum = diffcheckerPBA[np.absolute(diffcheckerPBA) <= threshold].sum().sum()

# Add the below-threshold sum as a new row
filtered_df.loc[('Below Threshold', 'Sum of below threshold'), :] = below_threshold_sum

# Sort the DataFrame to keep the new row at the end (optional)
# filtered_df = filtered_df.sort_index()

# Choose a color palette (using Set1)
colors = plt.get_cmap('Set1').colors

# Plot the filtered DataFrame with adjusted size and legend placement
ax = filtered_df.unstack().plot(kind="bar", stacked=True, legend=False, figsize=(10, 6), color=colors)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(True)  # Add grid lines
ax.set_title(f'Filtered difference in PBA (Aluminium interventions- baseline) in {indicator}')
ax.set_ylabel(f"{indicator} ({unit})")
ax.set_xlabel('Regions')

# Show the plot
plt.show()

#%% Make a graph that includes the below threshold values so it doesnt dissapear out of the system 
threshold = 10

# Filter the DataFrame to include only values above the threshold
filtered_df = diffcheckerCBA[np.absolute(diffcheckerCBA) > threshold].dropna()

# Calculate the sum of values below the threshold
below_threshold_sum = diffcheckerCBA[np.absolute(diffcheckerCBA) <= threshold].sum().sum()

# Add the below-threshold sum as a new row
filtered_df.loc[('Below Threshold', 'Sum of below threshold'), :] = below_threshold_sum

# Sort the DataFrame to keep the new row at the end (optional)
# filtered_df = filtered_df.sort_index()

# Choose a color palette (using Set1)
colors = plt.get_cmap('Set1').colors

# Plot the filtered DataFrame with adjusted size and legend placement
ax = filtered_df.unstack().plot(kind="bar", stacked=True, legend=False, figsize=(10, 6), color=colors)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(True)  # Add grid lines
ax.set_title(f'Filtered difference in CBA (Aluminium interventions - baseline) in {indicator}')
ax.set_ylabel(f"{indicator} ({unit})")
ax.set_xlabel('Regions')

# Show the plot
plt.show()


#%%Make the CBA and PBA to also include the Sectoral resultion for further analysis only for scenario ST
I = np.identity(A_St.shape[0])
L = np.linalg.inv(I-A_St)
x = L @ Y_St.sum(axis=1)

x_inv = x.copy()
x_inv[x_inv!=0] = 1/x_inv[x_inv!=0]

F_indicator = F_sat_St.loc[indicator]
F_hh_indicator = F_sat_hh.loc[indicator]

# calculate intensities
f_indicator = F_indicator @ np.diag(x_inv)
Y_reg = Y_St.groupby(level=0, axis=1, sort=False).sum()

#Consumption modeling baseline
CBA_St = f_indicator @ L @ np.diag(Y_reg.sum(axis =1)) #+ F_hh_indicator.groupby(level=0, axis=0, sort=False).sum()
#CBA_full.sort_values().iloc[[0, -1]]
CBA_St = pd.DataFrame(CBA_St, index = A.index)

#PBA calculations 
PBA_St = np.diag(f_indicator) @ x
PBA_St = pd.DataFrame(PBA_St)
PBA_St.index = A.index


diffcheckerCBA = CBA_St - CBA_baseline
diffcheckerPBA = PBA_St - PBA_baseline
#%% Make a graph that includes the below threshold values so it doesnt dissapear out of the system 
threshold = 0.5
# Filter the DataFrame to include only values above the threshold
filtered_df = diffcheckerPBA[np.absolute(diffcheckerPBA) > threshold].dropna()

# Calculate the sum of values below the threshold
below_threshold_sum = diffcheckerPBA[np.absolute(diffcheckerPBA) <= threshold].sum().sum()

# Add the below-threshold sum as a new row
filtered_df.loc[('Below Threshold', 'Sum of below threshold'), :] = below_threshold_sum

# Sort the DataFrame to keep the new row at the end (optional)
# filtered_df = filtered_df.sort_index()

# Choose a color palette (using Set1)
colors = plt.get_cmap('Set1').colors

# Plot the filtered DataFrame with adjusted size and legend placement
ax = filtered_df.unstack().plot(kind="bar", stacked=True, legend=False, figsize=(10, 6), color=colors)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(True)  # Add grid lines
ax.set_title(f'Filtered difference in PBA (Steel interventions- baseline) in {indicator}')
ax.set_ylabel(f"{indicator} ({unit})")
ax.set_xlabel('Regions')

# Show the plot
plt.show()

#%% Make a graph that includes the below threshold values so it doesnt dissapear out of the system 
threshold = 15

# Filter the DataFrame to include only values above the threshold
filtered_df = diffcheckerCBA[np.absolute(diffcheckerCBA) > threshold].dropna()

# Calculate the sum of values below the threshold
below_threshold_sum = diffcheckerCBA[np.absolute(diffcheckerCBA) <= threshold].sum().sum()

# Add the below-threshold sum as a new row
filtered_df.loc[('Below Threshold', 'Sum of below threshold'), :] = below_threshold_sum

# Sort the DataFrame to keep the new row at the end (optional)
# filtered_df = filtered_df.sort_index()

# Choose a color palette (using Set1)
colors = plt.get_cmap('Set1').colors

# Plot the filtered DataFrame with adjusted size and legend placement
ax = filtered_df.unstack().plot(kind="bar", stacked=True, legend=False, figsize=(10, 6), color=colors)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(True)  # Add grid lines
ax.set_title(f'Filtered difference in CBA (Steel - baseline) in {indicator}')
ax.set_ylabel(f"{indicator} ({unit})")
ax.set_xlabel('Regions')

# Show the plot
plt.show()

