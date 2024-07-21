# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:30:27 2024

@author: regin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
# =============================================================================
# Functions
# =============================================================================
def apply_shocks(file_path, A_matrix, Y_matrix, sequencesA, sequencesY, indicatorimpact, indicatorintensity, threshold,sensitivity,indicator):
    """

    Modify the A and Y matrix using an excel file with input parameters

    Creates a dataframe (same shape as the original dataframe) containing the modified values for both the 
    A and Y matrix based on the input parameters. 
    does also include system to perform different parameters

    Parameters:
        File_path (string):
            Dataframe containing results of Man-Kendall results (trend, p, intercept and slope)from mk function from  pymannkendall. 

        A_matrix (Dataframe_like)
            Baseline A matrix.  
            
        Y_matrix (Dataframe_like)
            Baseline Y matrix.

        sequencesA (Lists)
            List of lists containing the sequence of interventions implemented for each scenario  

        sequencesY (Lists)
            List of lists containing the sequence of interventions implemented for each scenario  
            
        indicatorimpact (Dataframe_like)
            Dataframe containing the environmental impact
            
        indicatorintensity (Dataframe_like)
            Dataframe environmental intensity 
        
        Threshold (float)
            Float of the passable threshold for visualization 
            
        Sensitivity (float)
            Extra input for changing the input of each intervention (Sensitivity = 1 when no sensitivity is applied)
            
        indicator (string)
            Enviromental impact of interest. 
            
    Returns: df_difference_output (Dictionary)
        Difference of gross output
        
    Returns:  df_difference_impact (Dictionary)
        Difference of environmental impact of interest

    Raises:
        IOError: An error occurred about the size of the model.
        
    """
    
    # Read the Excel file
    Full_shocks_A = pd.read_excel(file_path, sheet_name='z')
    Full_shocks_Y = pd.read_excel(file_path, sheet_name='Y')

    # Dictionary to store results for each sequence of interventions
    df_difference_output = {}
    df_difference_impact = {}

    # Ensure sequenceA and sequenceY are the same length
    max_length = max(len(sequencesA), len(sequencesY))
    sequencesA.extend([[]] * (max_length - len(sequencesA)))
    sequencesY.extend([[]] * (max_length - len(sequencesY)))

    # Process sequences together
    for seq_index, (seqA, seqY) in enumerate(zip(sequencesA, sequencesY)):
        # Create copies of the original DataFrames to modify
        A_modify = A_matrix.copy()
        Y_modify = Y_matrix.copy()

        # Process sequence A and the interventions for the A matrix
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

        # Process sequence Y and the interventions for the Y matrix
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

        Results_output = pd.DataFrame()
        Results_output["baseline"] = x_baseline
        Results_output["changes"] = x_modify
        Results_output["diff"] = Results_output["changes"] - Results_output["baseline"]
        Difference_output = Results_output["diff"]

        # Store the Results_output in the dictionary with the sequence index as the key
        df_difference_output[f'intervention{seq_index}'] = Difference_output
        df_difference_output = pd.DataFrame.from_dict(df_difference_output)

        L_ct = np.linalg.inv(I - A_modify.values)
        x_ct = L_ct @ Y_modify.values.sum(axis=1)
        RE_ct = indicatorintensity * x_ct
        F_diff_RE = (RE_ct - indicatorimpact)
        df_difference_impact[f'intervention{seq_index}'] = F_diff_RE
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

    return df_difference_output, df_difference_impact

def insert_break_string(text, threshold):
    """
    Add a line break for labels when longer than threshold. usefull when the labels are too long in graphs 
    Possible line of code, modified_labels = [insert_break_string(label) for label in labels]

    
    Parameters:
        text (Dataframe_like)
            Dataframe containing results of the simulated data.

        threshold (integer)
            number of letters before break is applied 
        

    Returns: List
        A list containing all labels with line break 

    Raises:
        IOError: An error occurred accessing the bigtable.Table object.
    """
    if len(text) <= threshold:
        return text
    space_index = text.find(' ', threshold)
    if space_index == -1:
        return text  # No space found after the 40th character, return as is.
    return text[:space_index] + '\n' + text[space_index+1:]
