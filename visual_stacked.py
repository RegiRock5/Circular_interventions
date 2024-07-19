# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:02:57 2024

@author: regin
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
inputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/output_visuals/"
hybrid_column = ["hyb_Al_scenario","hyb_Steel_scenario","hyb_combined_scenario"]
monetary_column = ["mon_Al_scenario","mon_Steel_scenario","mon_combined_scenario"]

unit = "Kilo tonnes"
co_unit = "Mega tonnes"
legend_size = 20
fontfont = 20
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

#%%Aluminium

# Define the threshold
threshold = 2
unit = "kt"  # Replace with the actual unit
fontfont = 20  # Adjust font size as needed
legend_size = 20  # Adjust legend size as needed

# Define the filtering and summarizing function
def filter_and_summarize(df, threshold):
    filtered_df = df[np.abs(df) > threshold].dropna()
    below_threshold_sum = df[np.abs(df) <= threshold].sum().sum()
    filtered_df.loc['Below Threshold'] = below_threshold_sum
    return filtered_df

# Filter and summarize the DataFrame
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
combined_df['Hybrid aluminium scenario'] = filtered_df1.reindex(combined_index).fillna(0)
combined_df['Hybrid steel scenario'] = filtered_df2.reindex(combined_index).fillna(0)
combined_df['Hybrid combined scenario'] = filtered_df3.reindex(combined_index).fillna(0)
combined_df['Monetary aluminium scenario'] = filtered_df4.reindex(combined_index).fillna(0)
combined_df['Monetary steel scenario'] = filtered_df5.reindex(combined_index).fillna(0)
combined_df['Monetary combined scenario'] = filtered_df6.reindex(combined_index).fillna(0)

combined_df = combined_df.T
combined_index = combined_df.index

# Create subplots
fig, ax = plt.subplots(figsize=(30, 16))
plt.rcParams.update({'font.size': fontfont})  # Adjust font size

colors = plt.get_cmap('tab20').colors
# Plot the stacked bar graph
combined_df.plot(kind='barh', stacked=True, ax=ax, color=colors)

# Set labels and title
ax.set_yticks(np.arange(len(combined_index)))
ax.set_yticklabels(combined_index)
ax.set_xlabel(f'Domestic extraction of bauxite and aluminium ore ({unit})',labelpad=30)
# ax.set_ylabel('Regions')

# Add legend
ax.legend(loc='best', fontsize=legend_size)

# Add grid lines
ax.xaxis.set_major_locator(plt.MultipleLocator(10))
# ax.grid(True, which='both', axis='y', linewidth=0.5)
ax.grid(True, which='both', axis='x', linewidth=1)

# Add a vertical line at zero for reference
plt.axvline(x=0, color='black', linewidth=1)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

print(combined_df.sum(0))

#%%iron

# Define the threshold
threshold = 5
unit = "kt"  # Replace with the actual unit
fontfont = 20  # Adjust font size as needed
legend_size = 20  # Adjust legend size as needed

# Filter and summarize the DataFrame
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
combined_df['Hybrid aluminium scenario'] = filtered_df1.reindex(combined_index).fillna(0)
combined_df['Hybrid steel scenario'] = filtered_df2.reindex(combined_index).fillna(0)
combined_df['Hybrid combined scenario'] = filtered_df3.reindex(combined_index).fillna(0)
combined_df['Monetary aluminium scenario'] = filtered_df4.reindex(combined_index).fillna(0)
combined_df['Monetary steel scenario'] = filtered_df5.reindex(combined_index).fillna(0)
combined_df['Monetary combined scenario'] = filtered_df6.reindex(combined_index).fillna(0)

combined_df = combined_df.T
combined_index = combined_df.index

# Create subplots
fig, ax = plt.subplots(figsize=(30, 16))
plt.rcParams.update({'font.size': fontfont})  # Adjust font size

colors = plt.get_cmap('tab20').colors
# Plot the stacked bar graph
combined_df.plot(kind='barh', stacked=True, ax=ax, color=colors)

# Set labels and title
ax.set_yticks(np.arange(len(combined_index)))
ax.set_yticklabels(combined_index)
ax.set_xlabel(f'Domestic extraction of iron ore ({unit})',labelpad=30)
# ax.set_ylabel('Regions')

# Add legend
ax.legend(loc='best', fontsize=legend_size)

# Add grid lines
ax.xaxis.set_major_locator(plt.MultipleLocator(100))
# ax.grid(True, which='both', axis='y', linewidth=0.5)
ax.grid(True, which='both', axis='x', linewidth=1)

# Add a vertical line at zero for reference
plt.axvline(x=0, color='black', linewidth=1)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

print(combined_df.sum(0))

#%%co2
# Define the threshold
threshold = 10
unit = "Mt CO2"  # Replace with the actual unit
fontfont = 20  # Adjust font size as needed
legend_size = 20  # Adjust legend size as needed

# Filter and summarize the DataFrame
CO_merged_df1 = CO_merged_df / 1000  # Assuming this conversion is needed

filtered_df1 = filter_and_summarize(CO_merged_df1['hyb_Al_scenario'], threshold)
filtered_df2 = filter_and_summarize(CO_merged_df1['hyb_Steel_scenario'], threshold)
filtered_df3 = filter_and_summarize(CO_merged_df1['hyb_combined_scenario'], threshold)

filtered_df4 = filter_and_summarize(CO_merged_df1['mon_Al_scenario'], threshold)
filtered_df5 = filter_and_summarize(CO_merged_df1['mon_Steel_scenario'], threshold)
filtered_df6 = filter_and_summarize(CO_merged_df1['mon_combined_scenario'], threshold)

# Combine the indexes of both filtered dataframes
combined_index = sorted(set(filtered_df1.index).union(set(filtered_df4.index)))

# Create a DataFrame with the combined index
combined_df = pd.DataFrame(index=combined_index)
combined_df['Hybrid aluminium scenario'] = filtered_df1.reindex(combined_index).fillna(0)
combined_df['Hybrid steel scenario'] = filtered_df2.reindex(combined_index).fillna(0)
combined_df['Hybrid combined scenario'] = filtered_df3.reindex(combined_index).fillna(0)
combined_df['Monetary aluminium scenario'] = filtered_df4.reindex(combined_index).fillna(0)
combined_df['Monetary steel scenario'] = filtered_df5.reindex(combined_index).fillna(0)
combined_df['Monetary combined scenario'] = filtered_df6.reindex(combined_index).fillna(0)

combined_df = combined_df.T
combined_index = combined_df.index

# Create subplots
fig, ax = plt.subplots(figsize=(30, 16))
plt.rcParams.update({'font.size': fontfont})  # Adjust font size

colors = plt.get_cmap('tab20').colors
# Plot the stacked bar graph
combined_df.plot(kind='barh', stacked=True, ax=ax, color=colors)

# Set labels and title
ax.set_yticks(np.arange(len(combined_index)))
ax.set_yticklabels(combined_index)
ax.set_xlabel(f'CO2 eq. ({unit})',labelpad=30)
# ax.set_ylabel('Regions')

# Add legend
ax.legend(loc='best', fontsize=legend_size)

# Add grid lines
ax.xaxis.set_major_locator(plt.MultipleLocator(50))
# ax.grid(True, which='both', axis='y', linewidth=0.5)
ax.grid(True, which='both', axis='x', linewidth=1)

# Add a vertical line at zero for reference
plt.axvline(x=0, color='black', linewidth=1)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

print(combined_df.sum(0))
#%% aluminium sens

# Define the threshold
threshold = 0.3
unit = "kt"  # Replace with the actual unit
fontfont = 20  # Adjust font size as needed
legend_size = 20  # Adjust legend size as needed
bar_width = 0.4


# Grouping data by index
AluminiumSenshyb = AluminiumSenshyb.groupby(level=0, axis=0, sort=False).sum()
AluminiumSensmon = AluminiumSensmon.groupby(level=0, axis=0, sort=False).sum()

# Define the filtering and summarizing function
def filter_and_summarize(df, threshold):
    filtered_df = df[np.abs(df) > threshold].dropna()
    below_threshold_sum = df[np.abs(df) <= threshold].sum().sum()
    filtered_df.loc['Below Threshold'] = below_threshold_sum
    return filtered_df

# Filter and summarize the DataFrame
filtered_df1 = filter_and_summarize(AluminiumSenshyb['increase_5%'], threshold)
filtered_df2 = filter_and_summarize(AluminiumSenshyb['increase_2%'], threshold)
filtered_df3 = filter_and_summarize(AluminiumSensmon['increase_5%'], threshold)
filtered_df4 = filter_and_summarize(AluminiumSensmon['increase_2%'], threshold)

# Combine the indexes of both filtered dataframes
combined_index = sorted(set(filtered_df1.index).union(set(filtered_df4.index)))

# Create a DataFrame with the combined index
combined_df = pd.DataFrame(index=combined_index)
combined_df['Hybrid 5% increase'] = filtered_df1.reindex(combined_index).fillna(0)
combined_df['Hybrid 2% increase'] = filtered_df2.reindex(combined_index).fillna(0)
combined_df['Monetary 5% increase'] = filtered_df3.reindex(combined_index).fillna(0)
combined_df['Monetary 2% increase'] = filtered_df4.reindex(combined_index).fillna(0)

combined_df = combined_df.T
combined_index = combined_df.index

# Create subplots
fig, ax = plt.subplots(figsize=(30, 14))
plt.rcParams.update({'font.size': fontfont})  # Adjust font size

colors = plt.get_cmap('tab20').colors
# Plot the stacked bar graph
combined_df.plot(kind='barh', stacked=True, ax=ax, color=colors, width = bar_width)

# Set labels and title
ax.set_yticks(np.arange(len(combined_index)))
ax.set_yticklabels(combined_index)
ax.set_xlabel(f'Domestic extraction of bauxite and aluminium ore ({unit})', labelpad=30)
# ax.set_ylabel('Regions')

# Add legend
ax.legend(loc='best', fontsize=legend_size)

# Add grid lines
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
# ax.grid(True, which='both', axis='y', linewidth=0.5)
ax.grid(True, which='both', axis='x', linewidth=1)

# Add a vertical line at zero for reference
plt.axvline(x=0, color='black', linewidth=1)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

print(combined_df.sum(0))


#%%iron sens
# Define the threshold
threshold = 0.5
unit = "kt"  # Replace with the actual unit
fontfont = 20  # Adjust font size as needed
legend_size = 20  # Adjust legend size as needed

# Grouping data by index
ironSenshyb = ironSenshyb.groupby(level=0, axis=0, sort=False).sum()
ironSensmon = ironSensmon.groupby(level=0, axis=0, sort=False).sum()

# Filter and summarize the DataFrame
filtered_df1 = filter_and_summarize(ironSenshyb['increase_5%'], threshold)
filtered_df2 = filter_and_summarize(ironSenshyb['increase_2%'], threshold)
filtered_df3 = filter_and_summarize(ironSensmon['increase_5%'], threshold)
filtered_df4 = filter_and_summarize(ironSensmon['increase_2%'], threshold)

# Combine the indexes of both filtered dataframes
combined_index = sorted(set(filtered_df1.index).union(set(filtered_df4.index)))

# Create a DataFrame with the combined index
combined_df = pd.DataFrame(index=combined_index)
combined_df['Hybrid 5% increase'] = filtered_df1.reindex(combined_index).fillna(0)
combined_df['Hybrid 2% increase'] = filtered_df2.reindex(combined_index).fillna(0)
combined_df['Monetary 5% increase'] = filtered_df3.reindex(combined_index).fillna(0)
combined_df['Monetary 2% increase'] = filtered_df4.reindex(combined_index).fillna(0)

combined_df = combined_df.T
combined_index = combined_df.index

# Create subplots
fig, ax = plt.subplots(figsize=(30, 14))
plt.rcParams.update({'font.size': fontfont})  # Adjust font size

colors = plt.get_cmap('tab20').colors
# Plot the stacked bar graph
combined_df.plot(kind='barh', stacked=True, ax=ax, color=colors, width = bar_width)

# Set labels and title
ax.set_yticks(np.arange(len(combined_index)))
ax.set_yticklabels(combined_index)
ax.set_xlabel(f'Domestic extraction of iron ore ({unit})',labelpad=30)
# ax.set_ylabel('Regions')

# Add legend
ax.legend(loc='best', fontsize=legend_size)

# Add grid lines
ax.xaxis.set_major_locator(plt.MultipleLocator(5))
# ax.grid(True, which='both', axis='y', linewidth=0.5)
ax.grid(True, which='both', axis='x', linewidth=1)

# Add a vertical line at zero for reference
plt.axvline(x=0, color='black', linewidth=1)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

print(combined_df.sum(0))

#%% co2 sens
# Define the threshold
threshold = 5
unit = "kt CO2"  # Replace with the actual unit
fontfont = 20  # Adjust font size as needed
legend_size = 20  # Adjust legend size as needed
#bar_width = 0.4

# Grouping data by index
CO2Senshyb = CO2Senshyb.groupby(level=0, axis=0, sort=False).sum()
CO2Sensmon = CO2Sensmon.groupby(level=0, axis=0, sort=False).sum()

# Filter and summarize the DataFrame
filtered_df1 = filter_and_summarize(CO2Senshyb['increase_5%'], threshold)
filtered_df2 = filter_and_summarize(CO2Senshyb['increase_2%'], threshold)
filtered_df3 = filter_and_summarize(CO2Sensmon['increase_5%'], threshold)
filtered_df4 = filter_and_summarize(CO2Sensmon['increase_2%'], threshold)

# Combine the indexes of both filtered dataframes
combined_index = sorted(set(filtered_df1.index).union(set(filtered_df4.index)))

# Create a DataFrame with the combined index
combined_df = pd.DataFrame(index=combined_index)
combined_df['Hybrid 5% increase'] = filtered_df1.reindex(combined_index).fillna(0)
combined_df['Hybrid 2% increase'] = filtered_df2.reindex(combined_index).fillna(0)
combined_df['Monetary 5% increase'] = filtered_df3.reindex(combined_index).fillna(0)
combined_df['Monetary 2% increase'] = filtered_df4.reindex(combined_index).fillna(0)

combined_df = combined_df.T
combined_index = combined_df.index

# Create subplots
fig, ax = plt.subplots(figsize=(30, 14))
plt.rcParams.update({'font.size': fontfont})  # Adjust font size

colors = plt.get_cmap('tab20').colors
# Plot the stacked bar graph
combined_df.plot(kind='barh', stacked=True, ax=ax, color=colors, width = bar_width)

# Set labels and title
ax.set_yticks(np.arange(len(combined_index)))
ax.set_yticklabels(combined_index)
ax.set_xlabel(f'Carbon footprint ({unit})',labelpad=30)
# ax.set_ylabel('Regions')

# Add legend
ax.legend(loc='best', fontsize=legend_size)

# Add grid lines
ax.xaxis.set_major_locator(plt.MultipleLocator(25))
# ax.grid(True, which='both', axis='y', linewidth=0.5)
ax.grid(True, which='both', axis='x', linewidth=1)

# Add a vertical line at zero for reference
plt.axvline(x=0, color='black', linewidth=1)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

print(combined_df.sum(1))
print(combined_df.sum().sum())

print(combined_df.NL.sum(1))

#%%
