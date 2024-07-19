# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:24:20 2024

@author: regin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
inputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/output_visuals/projection/version2/"

#%%
labelresource = "Bauxite_and_aluminium_ores"
aluminiummon = pd.read_csv(f'{inputpath}{labelresource}_mon_2050.csv', index_col=[0], header=[0])  
aluminiummon = aluminiummon.drop(columns=["sector"])
aluminiumhyb = pd.read_csv(f'{inputpath}{labelresource}_hyb_2050.csv', index_col=[0], header=[0])  

labelresource = "Iron_ores"
ironmon = pd.read_csv(f'{inputpath}{labelresource}_mon_2050.csv', index_col=[0], header=[0])  
ironmon = ironmon.drop(columns=["sector"])

ironhyb = pd.read_csv(f'{inputpath}{labelresource}_hyb_2050.csv', index_col=[0], header=[0])  

# labelresource = "CO2"
# CO2mon = pd.read_csv(f'{inputpath}{labelresource}_mon_2050.csv', index_col=[0], header=[0])  
# CO2hyb = pd.read_csv(f'{inputpath}{labelresource}_hyb_2050.csv', index_col=[0], header=[0])  

#%%
threshold = 10
unit = "kt"  # Replace with the actual unit
fontfont = 20  # Adjust font size as needed
legend_size = 20  # Adjust legend size as needed

# Define the filtering and summarizing function
def filter_and_summarize(df, threshold):
    filtered_df = df[np.abs(df) > threshold].dropna()
    below_threshold_sum = df[np.abs(df) <= threshold].sum().sum()
    filtered_df.loc['Below Threshold'] = below_threshold_sum
    return filtered_df

# # Filter and summarize the DataFrame
# filtered_df1 = filter_and_summarize(aluminiummon, threshold)
# filtered_df2 = filter_and_summarize(aluminiumhyb, threshold)


# Combine the indexes of both filtered dataframes
combined_index = sorted(set(aluminiummon.index).union(set(aluminiumhyb.index)))

# Create a DataFrame with the combined index
combined_df = pd.DataFrame(index=combined_index)


#%%
fontsize = 25
colorsfe = ['#E56134', '#F07E59'] #iron
colorsal = ['#47A690', '#59F0CC'] # aluminium
unit = "Mt"
# Number of bars
n_bars = len(combined_index)
bar_width = 0.33

combined_index = sorted(set(aluminiummon.index).union(set(aluminiumhyb.index)))
# Create a DataFrame with the combined index
combined_df = pd.DataFrame(index=combined_index)

# Create subplots
fig, ax = plt.subplots(figsize=(30, 16))
plt.rcParams.update({'font.size': fontsize})  # Reducing font size

# Calculate bar positions
spacing_factor = 2  # Increase to add more space between bars
r = np.arange(len(combined_index)) * spacing_factor
#r = np.arange(len(combined_index))


colors = plt.get_cmap('Set1').colors
cmap = plt.get_cmap("Paired")
colors = [cmap(i) for i in range(6)]

combined_df['Monetary 2050 material demand'] = aluminiummon.reindex(combined_index).fillna(0)
combined_df['Hybrid 2050 material demand'] = aluminiumhyb.reindex(combined_index).fillna(0)
combined_df = combined_df.sort_values(by= ["Monetary 2050 material demand"])
# Plot the data
bars1 = ax.bar(r, combined_df['Monetary 2050 material demand'], width=bar_width, label='Monetary 2050 material demand', color= colorsal[0])
bars2 = ax.bar(r +  bar_width, combined_df['Hybrid 2050 material demand'], width=bar_width, label='Hybrid 2050 material demand', color= colorsal[1])

bars1[4].set_color('#B0B0B0')
bars2[4].set_color('#C4C4C4')


# Set labels and title
ax.set_xticks(r + bar_width / 2)
ax.set_ylim(0, combined_df.max().max() * 1.2)
ax.set_xticklabels(combined_df.index, fontsize = fontsize)
ax.set_ylabel(f'Domestic extraction of Bauxite and Aluminium ore ({unit})',fontsize = fontsize, labelpad=20)
ax.set_xlabel('Regions')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

for p in ax.patches:
    height = p.get_height()
    if height != 0:
        ax.annotate(f'{height:.1f}{unit}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', fontsize=24, rotation = 90, xytext=(0, 70), textcoords='offset points')


# Add legend
ax.legend()
ax.legend(loc='upper left',fontsize = fontsize)

# Add grid lines
ax.yaxis.set_major_locator(plt.MultipleLocator(5))
# ax.grid(True, which='both', axis='y', linewidth=0.5)
ax.grid(True, which='both', axis='y', linewidth=1)

# Add a vertical line at zero for reference

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

print(combined_df.sum(0))

#%%
combined_index = sorted(set(ironmon.index).union(set(ironhyb.index)))
# Create a DataFrame with the combined index
combined_df = pd.DataFrame(index=combined_index)

fontsize = 25
colorsfe = ['#E56134', '#EDDDC5'] #iron
colorsal = ['#47A690', '#59F0CC'] # aluminium
unit = "Mt"
# Number of bars
n_bars = len(combined_index)
bar_width = 0.33

# Create subplots
fig, ax = plt.subplots(figsize=(30, 16))
plt.rcParams.update({'font.size': fontsize})  # Reducing font size

# Calculate bar positions
spacing_factor = 2  # Increase to add more space between bars
r = np.arange(len(combined_index)) * spacing_factor
#r = np.arange(len(combined_index))


colors = plt.get_cmap('Set1').colors
cmap = plt.get_cmap("Paired")
colors = [cmap(i) for i in range(6)]

combined_df['Monetary 2050 material demand'] = ironmon.reindex(combined_index).fillna(0)
combined_df['Hybrid 2050 material demand'] = ironhyb.reindex(combined_index).fillna(0)
combined_df = combined_df.sort_values(by= ["Monetary 2050 material demand"])
# Plot the data
bars1 = ax.bar(r, combined_df['Monetary 2050 material demand'], width=bar_width, label='Monetary 2050 material demand', color= colorsfe[0])
bars2 = ax.bar(r +  bar_width, combined_df['Hybrid 2050 material demand'], width=bar_width, label='Hybrid 2050 material demand', color= colorsfe[1])

bars1[11].set_color('#B0B0B0')
bars2[11].set_color('#C4C4C4')


# Set labels and title
ax.set_xticks(r + bar_width / 2)
ax.set_ylim(0, combined_df.max().max() * 1.2)
ax.set_xticklabels(combined_df.index, fontsize = fontsize)
ax.set_ylabel(f'Domestic extraction of iron ore ({unit})',fontsize = fontsize, labelpad=20)
ax.set_xlabel('Regions')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

for p in ax.patches:
    height = p.get_height()
    if height != 0:
        ax.annotate(f'{height:.1f}{unit}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', fontsize=24, rotation = 90, xytext=(0, 70), textcoords='offset points')


# Add legend
ax.legend()
ax.legend(loc='upper left',fontsize = fontsize)

# Add grid lines
ax.yaxis.set_major_locator(plt.MultipleLocator(100))
# ax.grid(True, which='both', axis='y', linewidth=0.5)
ax.grid(True, which='both', axis='y', linewidth=1)

# Add a vertical line at zero for reference

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

print(combined_df.sum(0))


# Sample Data (replace with your actual data)
