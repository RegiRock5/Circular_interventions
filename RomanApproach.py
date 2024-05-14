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
current_pop = 17982825
project_pop = 20610000
Populationgrowth = project_pop / current_pop
#%%
path = r"C:/Industrial_ecology/Thesis/IOT_2021_ixi/"
outputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/outputdata/" 

#"C:\Industrial_ecology\Thesis\Circularinterventions\Code\Input_circular_interventions\newZ.csv"
Anew = pd.read_csv(f"{outputpath}newA2.csv", sep=',', header=[0, 2])
Y = pd.read_csv(f'{path}Y.txt' , sep='\t', index_col=[0, 1], header=[0, 1])
A = pd.read_csv(f'{path}A.txt', sep='\t', index_col=[0, 1], header=[0, 1])
Ynew = pd.read_csv(f'{outputpath}newY.csv' , sep=',', header=[0,2])

#%%
# Import satellite accounts
F_sat = pd.read_csv(f'{path}satellite/F.txt' , sep='\t', index_col=[0], header=[0, 1])
F_sat_hh = pd.read_csv(f'{path}satellite/F_hh.txt' , sep='\t', index_col=[0], header=[0, 1])
#%% Import impact accounts
F_imp = pd.read_csv(f'{path}impacts/F.txt' , sep='\t', index_col=[0], header=[0, 1])
F_imp_hh = pd.read_csv(f'{path}impacts/F_hh.txt' , sep='\t', index_col=[0], header=[0, 1])

#%% Make the Z matrix using the A and Y
I = np.identity(A.shape[0])
L = np.linalg.inv(I - A)
x_org = L @ Y.sum(axis=1)
Z_org = A @ (np.diag(x_org))

#%%population growth is based on increasing the final demadn to match the growing population and multiply it with the final dea
Y.NL = Y.NL * Populationgrowth
x = L @ Y
Z = A@(np.diag(x.sum(axis = 1)))
Z.columns = Z.index
gross_output = Z.sum(axis = 1) + Y.sum(axis = 1)


#%%  prepare necessary data
Va = gross_output - Z.sum(axis=1)

#%%Value added
Y_agg = Y.groupby(level=0, axis=1, sort=False).sum()
xdf = pd.DataFrame(x, index= Z.index, columns=["output orginal"])
Y_agg_org = Y_agg.copy()



#%% Apply changes in Final demand, Value added and Z 

Va_nl = Va.NL
Va_nl = Va_nl * TFP_change

Z_nl = Z.NL
Z_nl = Z_nl / TFP_change

Y_agg_nl = Y_agg.NL
Y_agg_nl = Y_agg_nl* TFP_change

#%%apply changes back into the full dataset
Va.NL = Va_nl.values
Z.NL= Z_nl
Y_agg.NL = Y_agg_nl

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
fig, axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(50, 22))
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
    ax.tick_params(rotation=90)
    ax.xaxis.set_tick_params(which='both')

# Adjust layout to prevent overlapping
plt.tight_layout(pad=3.0)
plt.xticks(range(0,len(sector_labels.index)), sector_labels.index)

# Show the plot
plt.show()

#%% plot the data 2
# Create a figure and subplots
fig, axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(70, 22))
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
axs[4].legend()

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