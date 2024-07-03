# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:52:17 2024

@author: regin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
inputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/output_visuals/"

hybrid_column = ["hyb_Al_scenario","hyb_Steel_scenario","hyb_combined_scenario"]
monetary_column = ["mon_Al_scenario","mon_Steel_scenario","mon_combined_scenario"]
#%%Monetary tables
indicator = "Domestic Extraction Used - Metal Ores - Bauxite and aluminium ores"
labelresource = indicator[40:]
labelresource = labelresource.replace(" ", "_")
outputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/output_visuals/"
AluminiumImpactmon = pd.read_csv(f'{inputpath}{labelresource}_mon_impact.csv', index_col=[0,1], header=[0])  
AluminiumSensmon = pd.read_csv(f'{inputpath}{labelresource}_mon_sens.csv', index_col = [0])
AluminiumSensmon.index = AluminiumImpactmon.index
AluminiumImpactmon.columns = monetary_column

indicator ="Domestic Extraction Used - Metal Ores - Iron ores"
labelresource = indicator[40:]
labelresource = labelresource.replace(" ", "_")
outputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/output_visuals/"
ironImpactmon = pd.read_csv(f'{inputpath}{labelresource}_mon_impact.csv' , index_col=[0,1], header=[0])  
ironSensmon = pd.read_csv(f'{inputpath}{labelresource}_mon_sens.csv', index_col = [0])
ironSensmon.index = ironImpactmon.index
ironImpactmon.columns = monetary_column

#%% Hybrid tables
resource = "Bauxite and aluminium ores"
labelresource = resource
labelresource = labelresource.replace(" ", "_")
resource = "Bauxite and aluminium ores"
outputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/output_visuals/"
AluminiumImpacthyb = pd.read_csv(f'{inputpath}{labelresource}_hyb_impact.csv', index_col=[0,1], header=[0])  
AluminiumSenshyb = pd.read_csv(f'{inputpath}{labelresource}_hyb_sens.csv', index_col = [0])
AluminiumSenshyb.index  = AluminiumImpacthyb.index
AluminiumImpacthyb.columns = hybrid_column

resource = "Iron ores"
labelresource = resource
labelresource = labelresource.replace(" ", "_")
outputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/output_visuals/"
ironImpacthyb = pd.read_csv(f'{inputpath}{labelresource}_hyb_impact.csv', index_col=[0,1], header=[0])  
ironSenshyb = pd.read_csv(f'{inputpath}{labelresource}_hyb_sens.csv', index_col = [0])
ironSenshyb.index = ironImpacthyb.index
ironImpacthyb.columns = hybrid_column

#%%CO@
labelresource = "CO2"
outputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/output_visuals/"
CO2Impactmon = pd.read_csv(f'{inputpath}{labelresource}_mon_impact.csv', index_col=[0,1], header=[0])  
CO2Sensmon = pd.read_csv(f'{inputpath}{labelresource}_mon_sens.csv', index_col = [0])
CO2Sensmon.index = CO2Impactmon.index
CO2Impactmon.columns = monetary_column

Emission = "Carbon dioxide, fossil"
labelresource = Emission
labelresource = labelresource[:14]
labelresource = labelresource.replace(" ", "_")
outputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/output_visuals/"
CO2Impacthyb = pd.read_csv(f'{inputpath}{labelresource}_hyb_impact.csv', index_col=[0,1], header=[0])  
CO2Senshyb = pd.read_csv(f'{inputpath}{labelresource}_hyb_sens.csv', index_col = [0])
CO2Senshyb.index = CO2Impacthyb.index
CO2Impacthyb.columns = hybrid_column

#%%
AluminiumImpactmon = AluminiumImpactmon.groupby(level=0, axis=0, sort=False).sum()
AluminiumImpacthyb = AluminiumImpacthyb.groupby(level=0, axis=0, sort=False).sum()
AluminiumImpacthyb.loc["TW"] = [0,0,0]
AL_merged_df = pd.merge(AluminiumImpactmon, AluminiumImpacthyb, left_index=True, right_index=True)
#%%
ironImpactmon = ironImpactmon.groupby(level=0, axis=0, sort=False).sum()
ironImpacthyb = ironImpacthyb.groupby(level=0, axis=0, sort=False).sum()
ironImpacthyb.loc["TW"] = [0,0,0]
ST_merged_df = pd.merge(ironImpactmon, ironImpacthyb, left_index=True, right_index=True)

#%%
CO2Impactmon = CO2Impactmon.groupby(level=0, axis=0, sort=False).sum()
CO2Impacthyb = CO2Impacthyb.groupby(level=0, axis=0, sort=False).sum()
CO2Impacthyb.loc["TW"] = [0,0,0]
CO_merged_df = pd.merge(CO2Impactmon, CO2Impacthyb, left_index=True, right_index=True)

#%% graph for ALuminium ore
threshold = 2
hybrid_column = ["hyb_Al_scenario","hyb_Steel_scenario","hyb_combined_scenario"]
monetary_column = ["mon_Al_scenario","mon_Steel_scenario","mon_combined_scenario"]

def filter_and_summarize(df, threshold):
    filtered_df = df[np.abs(df) > threshold].dropna()
    below_threshold_sum = df[np.abs(df) <= threshold].sum().sum()
    filtered_df.loc['Below Threshold'] = below_threshold_sum
    return filtered_df

# Filter and summarize both DataFrames
filtered_df1 = filter_and_summarize(AL_merged_df['hyb_Al_scenario'], threshold)
filtered_df2 = filter_and_summarize(AL_merged_df['hyb_Steel_scenario'], threshold)
filtered_df3 = filter_and_summarize(AL_merged_df['hyb_combined_scenario'], threshold)

filtered_df4 = filter_and_summarize(AL_merged_df['mon_Al_scenario'], threshold)
filtered_df5 = filter_and_summarize(AL_merged_df['mon_Steel_scenario'], threshold)
filtered_df6 = filter_and_summarize(AL_merged_df['mon_combined_scenario'], threshold)

# Combine the indexes of both filtered dataframes
combined_index = sorted(set(filtered_df1.index).union(set(filtered_df4.index)))

# Create a DataFrame with the combined index
combined_df = pd.DataFrame(index=combined_index)
combined_df['hyb_Al_scenario'] = filtered_df1.reindex(combined_index).fillna(0)
combined_df['hyb_Steel_scenario'] = filtered_df2.reindex(combined_index).fillna(0)
combined_df['hyb_combined_scenario'] = filtered_df3.reindex(combined_index).fillna(0)
combined_df['mon_Al_scenario'] = filtered_df4.reindex(combined_index).fillna(0)
combined_df['mon_Steel_scenario'] = filtered_df5.reindex(combined_index).fillna(0)
combined_df['mon_combined_scenario'] = filtered_df6.reindex(combined_index).fillna(0)

# Number of bars
n_bars = len(combined_index)
bar_width = 0.33

# Create subplots
fig, ax = plt.subplots(figsize=(30, 16))
plt.rcParams.update({'font.size': 18})  # Reducing font size

# Calculate bar positions
spacing_factor = 2  # Increase to add more space between bars
r = np.arange(len(combined_index)) * spacing_factor
#r = np.arange(len(combined_index))
rslash = r +(bar_width/4)
rslash2 = r - (bar_width/4)

colors = plt.get_cmap('Set1').colors
cmap = plt.get_cmap("Paired")
colors = [cmap(i) for i in range(6)]

# Plot the data
ax.barh(r + bar_width, combined_df['hyb_combined_scenario'], height=bar_width, label='Hybrid combined scenario', color= colors[0])
ax.barh(r + 2 * bar_width, combined_df['mon_combined_scenario'], height=bar_width, label='Monetary combined scenario', color= colors[1])
ax.barh(r + 3 * bar_width, combined_df['hyb_Al_scenario'], height=bar_width, label='Hybrid Al scenario', color= colors[2])
ax.barh(r + 4 * bar_width, combined_df['mon_Al_scenario'], height=bar_width, label='Monetary Al scenario', color= colors[3])
ax.barh(r + 5 * bar_width,combined_df['hyb_Steel_scenario'], height=bar_width, label='Hybrid Steel scenario', color= colors[4])
ax.barh(r + 6 * bar_width, combined_df['mon_Steel_scenario'], height=bar_width, label='Monetary Steel scenario', color= colors[5])


# Set labels and title
ax.set_yticks(r + bar_width / 2)
ax.set_yticklabels(combined_index, fontsize=12)
ax.set_xlabel('Domestic extraction of Bauxite and Aluminium ore (kilotonnes)')
ax.set_title('Difference between IOT and HIOT ')
ax.set_ylabel('Regions')

# Add legend
ax.legend()
ax.legend(loc='right', bbox_to_anchor=(1, 0.5),fontsize = 12)

# Add grid lines
ax.grid(True)

# Add a vertical line at zero for reference
plt.axvline(x=0, color='black', linewidth=0.5)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

#%%

threshold = 1

hybrid_column = ["hyb_Al_scenario","hyb_Steel_scenario","hyb_combined_scenario"]
monetary_column = ["mon_Al_scenario","mon_Steel_scenario","mon_combined_scenario"]

def filter_and_summarize(df, threshold):
    filtered_df = df[np.abs(df) > threshold].dropna()
    below_threshold_sum = df[np.abs(df) <= threshold].sum().sum()
    filtered_df.loc['Below Threshold'] = below_threshold_sum
    return filtered_df

# Filter and summarize both DataFrames
filtered_df1 = filter_and_summarize(ST_merged_df['hyb_Al_scenario'], threshold)
filtered_df2 = filter_and_summarize(ST_merged_df['hyb_Steel_scenario'], threshold)
filtered_df3 = filter_and_summarize(ST_merged_df['hyb_combined_scenario'], threshold)

filtered_df4 = filter_and_summarize(ST_merged_df['mon_Al_scenario'], threshold)
filtered_df5 = filter_and_summarize(ST_merged_df['mon_Steel_scenario'], threshold)
filtered_df6 = filter_and_summarize(ST_merged_df['mon_combined_scenario'], threshold)

# Combine the indexes of both filtered dataframes
combined_index = sorted(set(filtered_df1.index).union(set(filtered_df4.index)))

# Create a DataFrame with the combined index
combined_df = pd.DataFrame(index=combined_index)
combined_df['hyb_Al_scenario'] = filtered_df1.reindex(combined_index).fillna(0)
combined_df['hyb_Steel_scenario'] = filtered_df2.reindex(combined_index).fillna(0)
combined_df['hyb_combined_scenario'] = filtered_df3.reindex(combined_index).fillna(0)
combined_df['mon_Al_scenario'] = filtered_df4.reindex(combined_index).fillna(0)
combined_df['mon_Steel_scenario'] = filtered_df5.reindex(combined_index).fillna(0)
combined_df['mon_combined_scenario'] = filtered_df6.reindex(combined_index).fillna(0)

# Number of bars
n_bars = len(combined_index)
bar_width = 0.33

# Create subplots
fig, ax = plt.subplots(figsize=(30, 16))
plt.rcParams.update({'font.size': 18})  # Reducing font size

# Calculate bar positions
spacing_factor = 2  # Increase to add more space between bars
r = np.arange(len(combined_index)) * spacing_factor
#r = np.arange(len(combined_index))
rslash = r +(bar_width/4)
rslash2 = r - (bar_width/4)

cmap = plt.get_cmap("Paired")
colors = [cmap(i) for i in range(6)]

# Plot the data
ax.barh(r + bar_width, combined_df['hyb_combined_scenario'], height=bar_width, label='Hybrid combined scenario', color= colors[0])
ax.barh(r + 2 * bar_width, combined_df['mon_combined_scenario'], height=bar_width, label='Monetary combined scenario', color= colors[1])
ax.barh(r + 3 * bar_width, combined_df['hyb_Al_scenario'], height=bar_width, label='Hybrid Al scenario', color= colors[2])
ax.barh(r + 4 * bar_width, combined_df['mon_Al_scenario'], height=bar_width, label='Monetary Al scenario', color= colors[3])
ax.barh(r + 5 * bar_width,combined_df['hyb_Steel_scenario'], height=bar_width, label='Hybrid Steel scenario', color= colors[4])
ax.barh(r + 6 * bar_width, combined_df['mon_Steel_scenario'], height=bar_width, label='Monetary Steel scenario', color= colors[5])

# Set labels and title
ax.set_yticks(r+ bar_width /2)
ax.set_yticklabels(combined_index, fontsize=16)
ax.set_xlabel('Domestic extraction of Iron ore (kilotonnes)')
ax.set_title('Difference between IOT and HIOT')
ax.set_ylabel('Regions')


# Add legend
ax.legend(fontsize = 15)
#ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5),fontsize = 12)

# Add grid lines
ax.grid(True)

# Add a vertical line at zero for reference
plt.axvline(x=0, color='black', linewidth=1)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

#%%Graph for iron ore

threshold = 10

hybrid_column = ["hyb_Al_scenario","hyb_Steel_scenario","hyb_combined_scenario"]
monetary_column = ["mon_Al_scenario","mon_Steel_scenario","mon_combined_scenario"]

def filter_and_summarize(df, threshold):
    filtered_df = df[np.abs(df) > threshold].dropna()
    below_threshold_sum = df[np.abs(df) <= threshold].sum().sum()
    filtered_df.loc['Below Threshold'] = below_threshold_sum
    return filtered_df

# Filter and summarize both DataFrames
filtered_df1 = filter_and_summarize(CO_merged_df['hyb_Al_scenario'], threshold)
filtered_df2 = filter_and_summarize(CO_merged_df['hyb_Steel_scenario'], threshold)
filtered_df3 = filter_and_summarize(CO_merged_df['hyb_combined_scenario'], threshold)

filtered_df4 = filter_and_summarize(CO_merged_df['mon_Al_scenario'], threshold)
filtered_df5 = filter_and_summarize(CO_merged_df['mon_Steel_scenario'], threshold)
filtered_df6 = filter_and_summarize(CO_merged_df['mon_combined_scenario'], threshold)

# Combine the indexes of both filtered dataframes
combined_index = sorted(set(filtered_df1.index).union(set(filtered_df4.index)))

# Create a DataFrame with the combined index
combined_df = pd.DataFrame(index=combined_index)
combined_df['hyb_Al_scenario'] = filtered_df1.reindex(combined_index).fillna(0)
combined_df['hyb_Steel_scenario'] = filtered_df2.reindex(combined_index).fillna(0)
combined_df['hyb_combined_scenario'] = filtered_df3.reindex(combined_index).fillna(0)
combined_df['mon_Al_scenario'] = filtered_df4.reindex(combined_index).fillna(0)
combined_df['mon_Steel_scenario'] = filtered_df5.reindex(combined_index).fillna(0)
combined_df['mon_combined_scenario'] = filtered_df6.reindex(combined_index).fillna(0)

# Number of bars
n_bars = len(combined_index)
bar_width = 0.33

# Create subplots
fig, ax = plt.subplots(figsize=(30, 16))
plt.rcParams.update({'font.size': 18})  # Reducing font size

# Calculate bar positions
spacing_factor = 2  # Increase to add more space between bars
r = np.arange(len(combined_index)) * spacing_factor
#r = np.arange(len(combined_index))
rslash = r +(bar_width/4)
rslash2 = r - (bar_width/4)

cmap = plt.get_cmap("Paired")
colors = [cmap(i) for i in range(6)]

# Plot the data
ax.barh(r + bar_width, combined_df['hyb_combined_scenario'], height=bar_width, label='Hybrid combined scenario', color= colors[0])
ax.barh(r + 2 * bar_width, combined_df['mon_combined_scenario'], height=bar_width, label='Monetary combined scenario', color= colors[1])
ax.barh(r + 3 * bar_width, combined_df['hyb_Al_scenario'], height=bar_width, label='Hybrid Al scenario', color= colors[2])
ax.barh(r + 4 * bar_width, combined_df['mon_Al_scenario'], height=bar_width, label='Monetary Al scenario', color= colors[3])
ax.barh(r + 5 * bar_width,combined_df['hyb_Steel_scenario'], height=bar_width, label='Hybrid Steel scenario', color= colors[4])
ax.barh(r + 6 * bar_width, combined_df['mon_Steel_scenario'], height=bar_width, label='Monetary Steel scenario', color= colors[5])

# Set labels and title
ax.set_yticks(r+ bar_width /2)
ax.set_yticklabels(combined_index, fontsize=16)
ax.set_xlabel('CO2 eq. differences (kilotonnes)')
ax.set_title('Difference between IOT and HIOT')
ax.set_ylabel('Regions')


# Add legend
ax.legend(fontsize = 15)
#ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5),fontsize = 12)

# Add grid lines
ax.grid(True)

# Add a vertical line at zero for reference
plt.axvline(x=0, color='black', linewidth=1)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

#%%
#combined_df.loc["NL"].plot(kind= "bar")
#%% Sensitivity analysis on aluminium
threshold = 2
hybrid_column = ["hyb_Al_scenario","hyb_Steel_scenario","hyb_combined_scenario"]
monetary_column = ["mon_Al_scenario","mon_Steel_scenario","mon_combined_scenario"]

AluminiumSenshyb = AluminiumSenshyb.groupby(level=0, axis=0, sort=False).sum()
AluminiumSensmon = AluminiumSensmon.groupby(level=0, axis=0, sort=False).sum()

def filter_and_summarize(df, threshold):
    filtered_df = df[np.abs(df) > threshold].dropna()
    below_threshold_sum = df[np.abs(df) <= threshold].sum().sum()
    filtered_df.loc['Below Threshold'] = below_threshold_sum
    return filtered_df

# Filter and summarize both DataFrames
filtered_df1 = filter_and_summarize(AluminiumSenshyb['increase_5%'], threshold)
filtered_df2 = filter_and_summarize(AluminiumSenshyb['increase_2%'], threshold)

filtered_df3 = filter_and_summarize(AluminiumSensmon['increase_5%'], threshold)
filtered_df4 = filter_and_summarize(AluminiumSensmon['increase_2%'], threshold)

# Combine the indexes of both filtered dataframes
combined_index = sorted(set(filtered_df1.index).union(set(filtered_df4.index)))

# Create a DataFrame with the combined index
combined_df = pd.DataFrame(index=combined_index)
combined_df['hyb_increase_5%'] = filtered_df1.reindex(combined_index).fillna(0)
combined_df['hyb_increase_2%'] = filtered_df2.reindex(combined_index).fillna(0)
combined_df['mon_increase_5%'] = filtered_df3.reindex(combined_index).fillna(0)
combined_df['mon_increase_2%'] = filtered_df4.reindex(combined_index).fillna(0)


# Number of bars
n_bars = len(combined_index)
bar_width = 0.33

# Create subplots
fig, ax = plt.subplots(figsize=(30, 16))
plt.rcParams.update({'font.size': 18})  # Reducing font size

# Calculate bar positions
spacing_factor = 2  # Increase to add more space between bars
r = np.arange(len(combined_index)) * spacing_factor
#r = np.arange(len(combined_index))
rslash = r +(bar_width/4)
rslash2 = r - (bar_width/4)

colors = plt.get_cmap('Set1').colors
cmap = plt.get_cmap("Paired")
colors = [cmap(i) for i in range(6)]

# Plot the data
ax.barh(r + bar_width, combined_df['hyb_increase_5%'], height=bar_width, label='Hybrid 5% change in input', color= colors[0])
ax.barh(r + 2 * bar_width, combined_df['hyb_increase_2%'], height=bar_width, label='Hybrid 2% change in input', color= colors[2])
ax.barh(r + 3 * bar_width, combined_df['mon_increase_5%'], height=bar_width, label='Monetary 5% change in input', color= colors[1])
ax.barh(r + 4 * bar_width, combined_df['mon_increase_2%'], height=bar_width, label='Monetary 2% change in input', color= colors[3])



# Set labels and title
ax.set_yticks(r + bar_width / 2)
ax.set_yticklabels(combined_index, fontsize=12)
ax.set_xlabel('Domestic extraction of Bauxite and Aluminium ore (kilotonnes)')
ax.set_title('Sensitivity between IOT and HIOT')
ax.set_ylabel('Regions')

# Add legend
ax.legend()
ax.legend(loc='right', bbox_to_anchor=(1, 0.5),fontsize = 12)

# Add grid lines
ax.grid(True)

# Add a vertical line at zero for reference
plt.axvline(x=0, color='black', linewidth=0.5)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


#%% Sensitivity analysis on iron
threshold = 2
hybrid_column = ["hyb_Al_scenario","hyb_Steel_scenario","hyb_combined_scenario"]
monetary_column = ["mon_Al_scenario","mon_Steel_scenario","mon_combined_scenario"]

ironSenshyb = ironSenshyb.groupby(level=0, axis=0, sort=False).sum()
ironSensmon = ironSensmon.groupby(level=0, axis=0, sort=False).sum()


def filter_and_summarize(df, threshold):
    filtered_df = df[np.abs(df) > threshold].dropna()
    below_threshold_sum = df[np.abs(df) <= threshold].sum().sum()
    filtered_df.loc['Below Threshold'] = below_threshold_sum
    return filtered_df

# Filter and summarize both DataFrames
filtered_df1 = filter_and_summarize(ironSenshyb['increase_5%'], threshold)
filtered_df2 = filter_and_summarize(ironSenshyb['increase_2%'], threshold)

filtered_df3 = filter_and_summarize(ironSensmon['increase_5%'], threshold)
filtered_df4 = filter_and_summarize(ironSensmon['increase_2%'], threshold)

# Combine the indexes of both filtered dataframes
combined_index = sorted(set(filtered_df1.index).union(set(filtered_df4.index)))

# Create a DataFrame with the combined index
combined_df = pd.DataFrame(index=combined_index)
combined_df['hyb_increase_5%'] = filtered_df1.reindex(combined_index).fillna(0)
combined_df['hyb_increase_2%'] = filtered_df2.reindex(combined_index).fillna(0)
combined_df['mon_increase_5%'] = filtered_df3.reindex(combined_index).fillna(0)
combined_df['mon_increase_2%'] = filtered_df4.reindex(combined_index).fillna(0)


# Number of bars
n_bars = len(combined_index)
bar_width = 0.33

# Create subplots
fig, ax = plt.subplots(figsize=(30, 16))
plt.rcParams.update({'font.size': 18})  # Reducing font size

# Calculate bar positions
spacing_factor = 2  # Increase to add more space between bars
r = np.arange(len(combined_index)) * spacing_factor
#r = np.arange(len(combined_index))
rslash = r +(bar_width/4)
rslash2 = r - (bar_width/4)

colors = plt.get_cmap('Set1').colors
cmap = plt.get_cmap("Paired")
colors = [cmap(i) for i in range(6)]

# Plot the data
ax.barh(r + bar_width, combined_df['hyb_increase_5%'], height=bar_width, label='Hybrid 5% change in input', color= colors[0])
ax.barh(r + 2 * bar_width, combined_df['hyb_increase_2%'], height=bar_width, label='Hybrid 2% change in input', color= colors[2])
ax.barh(r + 3 * bar_width, combined_df['mon_increase_5%'], height=bar_width, label='Monetary 5% change in input', color= colors[1])
ax.barh(r + 4 * bar_width, combined_df['mon_increase_2%'], height=bar_width, label='Monetary 2% change in input', color= colors[3])



# Set labels and title
ax.set_yticks(r + bar_width / 2)
ax.set_yticklabels(combined_index, fontsize=12)
ax.set_xlabel('Domestic extraction of iron ore (kilotonnes)')
ax.set_title('Sensitivity between IOT and HIOT')
ax.set_ylabel('Regions')

# Add legend
ax.legend()
ax.legend(loc='right', bbox_to_anchor=(1, 0.5),fontsize = 12)

# Add grid lines
ax.grid(True)

# Add a vertical line at zero for reference
plt.axvline(x=0, color='black', linewidth=0.5)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


#%%

threshold = 2
hybrid_column = ["hyb_Al_scenario","hyb_Steel_scenario","hyb_combined_scenario"]
monetary_column = ["mon_Al_scenario","mon_Steel_scenario","mon_combined_scenario"]

CO2Senshyb = CO2Senshyb.groupby(level=0, axis=0, sort=False).sum()
CO2Sensmon = CO2Sensmon.groupby(level=0, axis=0, sort=False).sum()


def filter_and_summarize(df, threshold):
    filtered_df = df[np.abs(df) > threshold].dropna()
    below_threshold_sum = df[np.abs(df) <= threshold].sum().sum()
    filtered_df.loc['Below Threshold'] = below_threshold_sum
    return filtered_df

# Filter and summarize both DataFrames
filtered_df1 = filter_and_summarize(CO2Senshyb['increase_5%'], threshold)
filtered_df2 = filter_and_summarize(CO2Senshyb['increase_2%'], threshold)

filtered_df3 = filter_and_summarize(CO2Sensmon['increase_5%'], threshold)
filtered_df4 = filter_and_summarize(CO2Sensmon['increase_2%'], threshold)

# Combine the indexes of both filtered dataframes
combined_index = sorted(set(filtered_df1.index).union(set(filtered_df4.index)))

# Create a DataFrame with the combined index
combined_df = pd.DataFrame(index=combined_index)
combined_df['hyb_increase_5%'] = filtered_df1.reindex(combined_index).fillna(0)
combined_df['hyb_increase_2%'] = filtered_df2.reindex(combined_index).fillna(0)
combined_df['mon_increase_5%'] = filtered_df3.reindex(combined_index).fillna(0)
combined_df['mon_increase_2%'] = filtered_df4.reindex(combined_index).fillna(0)


# Number of bars
n_bars = len(combined_index)
bar_width = 0.33

# Create subplots
fig, ax = plt.subplots(figsize=(30, 16))
plt.rcParams.update({'font.size': 18})  # Reducing font size

# Calculate bar positions
spacing_factor = 2  # Increase to add more space between bars
r = np.arange(len(combined_index)) * spacing_factor
#r = np.arange(len(combined_index))
rslash = r +(bar_width/4)
rslash2 = r - (bar_width/4)

colors = plt.get_cmap('Set1').colors
cmap = plt.get_cmap("Paired")
colors = [cmap(i) for i in range(6)]

# Plot the data
ax.barh(r + bar_width, combined_df['hyb_increase_5%'], height=bar_width, label='Hybrid 5% change in input', color= colors[0])
ax.barh(r + 2 * bar_width, combined_df['hyb_increase_2%'], height=bar_width, label='Hybrid 2% change in input', color= colors[2])
ax.barh(r + 3 * bar_width, combined_df['mon_increase_5%'], height=bar_width, label='Monetary 5% change in input', color= colors[1])
ax.barh(r + 4 * bar_width, combined_df['mon_increase_2%'], height=bar_width, label='Monetary 2% change in input', color= colors[3])



# Set labels and title
ax.set_yticks(r + bar_width / 2)
ax.set_yticklabels(combined_index, fontsize=12)
ax.set_xlabel('CO2 footprint (kilotonnes)')
ax.set_title('Sensitivity between IOT and HIOT')
ax.set_ylabel('Regions')

# Add legend
ax.legend()
ax.legend(loc='right', bbox_to_anchor=(1, 0.5),fontsize = 12)

# Add grid lines
ax.grid(True)

# Add a vertical line at zero for reference
plt.axvline(x=0, color='black', linewidth=0.5)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
