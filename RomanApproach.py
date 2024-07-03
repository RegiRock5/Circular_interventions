# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:27:39 2024

@author: regin
"""

#import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Calcualte the Total factor productivity 
TFP_2010 = 0.54
TFP_growthrate = 1.055 
TFP_2050 = TFP_2010 * (TFP_growthrate**8)
TFP_change = TFP_2050/TFP_2010
Z_change = (1- TFP_2050) / (1 - TFP_2010) 
current_pop = 16655799
project_pop = 20610000
Populationgrowth = project_pop / current_pop
#%%
path = r"C:/Industrial_ecology/Thesis/IOT_2011_ixi/"
outputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/Code/Output/" 

#"C:\Industrial_ecology\Thesis\Circularinterventions\Code\Input_circular_interventions\newZ.csv"
Anew = pd.read_csv(f"{outputpath}A_full_adjusted.csv", sep=',', header=[0, 2])
Y = pd.read_csv(f'{path}Y.txt' , sep='\t', index_col=[0, 1], header=[0, 1])
A = pd.read_csv(f'{path}A.txt', sep='\t', index_col=[0, 1], header=[0, 1])
Ynew = pd.read_csv(f'{outputpath}Y_full_adjusted.csv' , sep=',', header=[0,2])

#%%
# Import satellite accounts
F_sat = pd.read_csv(f'{path}satellite/F.txt' , sep='\t', index_col=[0], header=[0, 1])
F_sat_hh = pd.read_csv(f'{path}satellite/F_hh.txt' , sep='\t', index_col=[0], header=[0, 1])
# #%% Import impact accounts
# F_imp = pd.read_csv(f'{path}impacts/F.txt' , sep='\t', index_col=[0], header=[0, 1])
# F_imp_hh = pd.read_csv(f'{path}impacts/F_hh.txt' , sep='\t', index_col=[0], header=[0, 1])

#%% Make the Z matrix using the A and Y
I = np.identity(A.shape[0])
L = np.linalg.inv(I - A)
x_org = L @ Y.sum(axis=1)
Z_org = A @ (np.diag(x_org))

# Z_org.NL.loc["NL"]

#%%population growth is based on increasing the final demadn to match the growing population and multiply it with the final dea
Y.NL = Y.NL * Populationgrowth
x = L @ Y
Z = A@(np.diag(x.sum(axis = 1)))
Z.columns = Z.index
gross_output = Z.sum(axis = 1) + Y.sum(axis = 1)

Z_popgrowth = Z.copy()

#%%  prepare necessary data
Va = gross_output - Z.sum(axis=1)

#%%Value added
Y_agg = Y.groupby(level=0, axis=1, sort=False).sum()
xdf = pd.DataFrame(x, index= Z.index, columns=["output orginal"])
Y_agg_org = Y_agg.copy()



#%% Apply changes in Final demand, Value added and Z 

Va_nl = Va#.NL
Va_nl = Va_nl / TFP_change

Z_nl = Z#.NL
Z_nl = Z_nl * TFP_change

Y_agg_nl = Y_agg#.NL
Y_agg_nl = Y_agg_nl / TFP_change

#%%apply changes back into the full dataset
# Va.NL = Va_nl.values
# Z.NL= Z_nl
# Y_agg.NL = Y_agg_nl
Va = Va_nl.values
Z= Z_nl
Y_agg = Y_agg_nl

new_input = Z.sum(axis = 1) + Va
newY = new_input - Z.sum(axis = 0)
new_output = Z.sum(axis = 0) + Y_agg.sum(axis =1)


newY.loc["NL"].plot()
Y_agg_org.NL.loc["NL"].plot()

difY = newY - Y_agg.sum(axis = 1)

changeY = Y_agg.NL.copy()
changY = changeY - difY

Z_diff = Z_org.values - Z.values
Z_diff = pd.DataFrame(Z_diff, index=Z.index)
Z_diff.columns = Z.index
Z_diff_nl = Z_diff.NL


#%% Changed Input output 

# x_new = Z.sum(axis=1) + Va 
# xdf["new input"] = x_new

# print(xdf.loc["NL"])
# xdfnl = xdf.loc["NL"]
# print(xdf.sum(axis = 0))


# new_final = x - Va
# new_final = pd.DataFrame(new_final)
# xdfnl.plot()

# #%%final check 
# #output

# #input
# new_input = Z.sum(axis = 1) + Va
# new_output = Z.sum(axis = 0) + Y_agg.sum(axis = 1)

resultsdf = pd.DataFrame()
resultsdf["output"] = new_output
resultsdf["input"] = new_input

print(resultsdf.sum(axis = 0))
resultsdf.plot()


print(new_input.sort_values())
print(new_output.sort_values())

print(resultsdf.sum(axis = 0))
#%%
resultsdf["popgrowth"] = gross_output
print(resultsdf.sum(axis = 0))
print(resultsdf.loc["NL"].sum(axis = 0))
resultsdf.loc["NL"].to_excel("wowow.xlsx")
resultsdf["org_output"] = x_org

resultsdf["difference_pop"] =  resultsdf.popgrowth - resultsdf.org_output 
resultsdf["difference_TFP"] = resultsdf.output - resultsdf.popgrowth


A_labels = A.index
A_labels = A_labels.to_frame(index=None)
sector_labels = A_labels.sector.drop_duplicates().reset_index(drop=True)
sector_labels = sector_labels.values
sector_labels = resultsdf.org_output.groupby(level=1, axis=0, sort=False).sum()
#%% plot the data 
# Create a figure and subplots
fig, axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(25, 22))
plt.rcParams.update({'font.size': 8})  # Reducing font size

# Plot the data on each subplot
resultsdf.org_output.loc["NL"].plot(ax=axs[0], label="Base gross output")
resultsdf.popgrowth.loc["NL"].plot(ax=axs[1], label="Population growth", color="red")
resultsdf.input.loc["NL"].plot(ax=axs[2], label="Input TFP", color="cyan")
resultsdf.output.loc["NL"].plot(ax=axs[2], label="Output TFP", color="magenta")
resultsdf.difference_pop.loc["NL"].plot(ax=axs[3], label="difference population growth", color="red")
resultsdf.difference_TFP.loc["NL"].plot(ax=axs[3], label="difference TFP output", color="green")


# Add legend to each subplot
axs[0].legend()
axs[1].legend()
axs[2].legend()
axs[3].legend()

for ax in axs:
    ax.legend()

tickvalues = range(0,len(sector_labels))

for ax in axs:
    ax.xaxis.set_tick_params(which='minor')
    ax.tick_params(rotation=90)

# Adjust layout to prevent overlapping
plt.tight_layout(pad=3.0)
plt.xticks(range(0,len(sector_labels.index)), sector_labels.index)

# Show the plot
plt.show()

#%% plot the data 2
# Create a figure and subplots
fig, axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(30, 22))
plt.rcParams.update({'font.size': 8})  # Reducing font size

# Plot the data on each subplot
resultsdf.org_output.groupby(level=0, axis=0, sort=False).sum().plot(ax=axs[0], label="Base gross output")
resultsdf.popgrowth.groupby(level=0, axis=0, sort=False).sum().plot(ax=axs[1], label="Population growth", color="red")
resultsdf.input.groupby(level=0, axis=0, sort=False).sum().plot(ax=axs[2], label="Input TFP", color="cyan")
resultsdf.output.groupby(level=0, axis=0, sort=False).sum().plot(ax=axs[2], label="Output TFP", color="magenta")
resultsdf.difference_pop.groupby(level=0, axis=0, sort=False).sum().plot(ax=axs[3], label="difference population growth", color="red")
resultsdf.difference_TFP.groupby(level=0, axis=0, sort=False).sum().plot(ax=axs[3], label="difference TFP output", color="green")

print(resultsdf.difference_pop.groupby(level=0, axis=0, sort=False).sum().sort_values(0))
print(resultsdf.difference_TFP.groupby(level=0, axis=0, sort=False).sum().sort_values(0))

# Add legend to each subplot
axs[0].legend()
axs[1].legend()
axs[2].legend()
axs[3].legend()

for ax in axs:
    ax.legend()

for ax in axs:
    ax.tick_params(axis='x', rotation=90)
    ax.xaxis.set_tick_params(which='both')

A_labels = A.index
A_labels = A_labels.to_frame(index=None)
sector_labels = A_labels.sector.drop_duplicates().reset_index(drop=True)

# Adjust layout to prevent overlapping
plt.tight_layout(pad=3.0)
plt.xticks(sector_labels)

# Show the plot
plt.show()

#%%

#%%
Z_org.columns = Z.index
# Z_org.NL.loc["NL"].plot()
# Z_popgrowth.NL.loc["NL"].plot()
Zdiffpop = Z_popgrowth.NL.loc["NL"] - Z_org.NL.loc["NL"] 
# Zdiffpop.sum(axis = 1).plot()


fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(30, 22))
plt.rcParams.update({'font.size': 20})  # Reducing font size

# Plot the data on each subplot
Zdiffpop.sum(axis = 0).plot(ax=axs[0], label="difference in output per industry")
Zdiffpop.sum(axis = 1).plot(ax=axs[1], label="difference in input per industry")


# Add legend to each subplot
axs[0].legend()
axs[1].legend()

    
tickvalues = range(0,len(sector_labels))

for ax in axs:
    ax.xaxis.set_tick_params(which='minor')
    ax.tick_params(rotation=90)

# Adjust layout to prevent overlapping
plt.tight_layout(pad=3.0)
plt.xticks(range(0,len(sector_labels.index)), sector_labels.index)

# Show the plot
plt.show()

#%%
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(30, 22))
plt.rcParams.update({'font.size': 20})  # Reducing font size

# Plot the data on each subplot
Zdiffpop.sum(axis=0).plot(ax=axs[0], label="difference in output per industry")
Zdiffpop.sum(axis=1).plot(ax=axs[1], label="difference in input per industry")

# Add legend to each subplot
axs[0].legend()
axs[1].legend()

# Define the threshold
threshold = 700  # Example threshold

# Determine which sectors meet the threshold
output_values = Zdiffpop.sum(axis=0)
input_values = Zdiffpop.sum(axis=1)

# Set tick values and labels based on the threshold
tickvalues = range(0, len(sector_labels))
filtered_labels = [label if output_values[i] > threshold or input_values[i] > threshold else '' for i, label in enumerate(sector_labels.index)]

# Apply x-axis labels conditionally
for ax in axs:
    ax.xaxis.set_tick_params(which='minor')
    ax.tick_params(rotation=90)
    ax.set_xticks(tickvalues)
    ax.set_xticklabels(filtered_labels)

# Adjust layout to prevent overlapping
plt.tight_layout(pad=3.0)

# Show the plot
plt.show()

#%% 
Zdiffpop.sum(axis=1).sort_values()

#%%
A_labels = A.index
A_labels = A_labels.to_frame(index=None)
region_labels = A_labels.region.drop_duplicates().reset_index(drop=True)
#%% plot the data 2
# Create a figure and subplots
fig, axs = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(30, 22))
plt.rcParams.update({'font.size': 20})  # Reducing font size

# Plot the data on each subplot
resultsdf.org_output.groupby(level=0, axis=0, sort=False).sum().plot(kind='bar',ax=axs[0], label="Base gross output", color="#2F94A8")
resultsdf.popgrowth.groupby(level=0, axis=0, sort=False).sum().plot(kind='bar',ax=axs[0], label="Population growth", color="#AD1556")
resultsdf.difference_pop.groupby(level=0, axis=0, sort=False).sum().plot(kind='bar',ax=axs[1], label="difference population growth", color="#512B84")
# Add legend to each subplot
axs[0].legend()
axs[1].legend()
axs[1].set_xticks(range(len(region_labels)))
axs[1].set_xticklabels(region_labels, rotation=90)

for ax in axs:
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel("In million euro's")
    ax.set_xlabel('Regions')
    ax.tick_params(axis='x', rotation=0)
    ax.xaxis.set_tick_params(which='both')
    ax.grid(True)  # Add grid lines
# Adjust layout to prevent overlapping
plt.tight_layout(pad=3.0)
plt.xticks(sector_labels)

# Show the plot
plt.show()

#%%
differencepop =  resultsdf.popgrowth - resultsdf.org_output 

#%% Make a graph that includes the below threshold values so it doesnt dissapear out of the system 
threshold = 4000

# Filter the DataFrame to include only values above the threshold
filtered_df = differencepop[np.absolute(differencepop) > threshold].dropna()

# Calculate the sum of values below the threshold
below_threshold_sum = differencepop[np.absolute(differencepop) <= threshold].sum().sum()

# Add the below-threshold sum as a new row
filtered_df[('Below Threshold', 'Sum of below threshold')] = below_threshold_sum

# Choose a color palette (using Set1)
colors = plt.get_cmap('Set1').colors

# Plot the filtered DataFrame with adjusted size and legend placement
ax = filtered_df.unstack().plot(kind="bar", stacked=True, legend=False, figsize=(10, 6), color=colors)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(True)  # Add grid lines
# ax.set_title(f'Filtered difference in CBA (Steel - baseline) in {indicator}')
# ax.set_ylabel(f"{indicator} ({unit})")
ax.set_xlabel('Regions')
plt.tight_layout(pad=3.0)

# Show the plot
plt.show()

#%%import pandas as pd

# Create a figure and subplots
fig, axs = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(30, 22))
plt.rcParams.update({'font.size': 20})  # Reducing font size

# Grouping and summing data for plotting
org_output = resultsdf.org_output.groupby(level=0, axis=0, sort=False).sum()
popgrowth = resultsdf.popgrowth.groupby(level=0, axis=0, sort=False).sum()
difference_pop = resultsdf.difference_pop.groupby(level=0, axis=0, sort=False).sum()

# Combine the data into DataFrames for side-by-side plotting
combined_df_1 = pd.DataFrame({'Gross output': org_output, 'Gross output 2050': popgrowth})
combined_df_2 = pd.DataFrame({'Delta gross output': difference_pop})

# Plot the data as bar graphs
combined_df_1.plot(kind='bar', ax=axs[0], color=["#2F94A8", "#AD1556"])
combined_df_2.plot(kind='bar', ax=axs[1], color=["#512B84"])

# Set titles for each subplot
axs[0].set_title('Gross output of 2011 and 2050 model')
axs[1].set_title('Difference due to population growth')

# Customize each subplot
for ax in axs:
    ax.legend(loc='best')
    ax.set_ylabel("Gross output (million euro's)")
    ax.grid(True)  # Add grid lines

# Set x-ticks and labels for both subplots
axs[1].set_xticks(range(len(region_labels)))
axs[1].set_xticklabels(region_labels, rotation=90)
axs[1].set_xlabel('Regions')

axs[0].set_xticks(range(len(region_labels)))
axs[0].set_xticklabels(region_labels, rotation=90)

# Adjust layout to prevent overlapping
plt.tight_layout(pad=4.0)

# Show the plot
plt.show()

