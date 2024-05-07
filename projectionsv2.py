# -*- coding: utf-8 -*-
"""
Created on Sun May  5 15:18:16 2024

@author: regin
"""
import numpy as np 
import pandas as pd
#%%
TFP_2010 = 0.54
TFP_growthrate = 1.055 
TFP_2050 = TFP_2010 * (TFP_growthrate**8)
TFP_change = TFP_2050/TFP_2010
Z_change = (1- TFP_2050) / (1 - TFP_2010) 
#%% Import data into the system
path = r"C:/Industrial_ecology/Thesis/IOT_2021_ixi/"
outputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/outputdata/" 

#"C:\Industrial_ecology\Thesis\Circularinterventions\Code\Input_circular_interventions\newZ.csv"
Anew = pd.read_csv(f"{outputpath}newA2.csv", sep=',', header=[0, 2])
Y = pd.read_csv(f'{path}Y.txt' , sep='\t', index_col=[0, 1], header=[0, 1])
A = pd.read_csv(f'{path}A.txt', sep='\t', index_col=[0, 1], header=[0, 1])
Ynew = pd.read_csv(f'{outputpath}newY.csv' , sep=',', header=[0,2])


#%% Make the Z matrix using the A and Y
I = np.identity(A.shape[0])
L = np.linalg.inv(I - A)
x = L @ Y.sum(axis=1)
Y_agg= Y.sum(axis = 1)

#%% Data preperation 
Z = A @ (np.diag(x))
Z.columns = Z.index

#%%
Value_added = x - Z.sum(axis=1)
Value_A_nl = Value_added.NL
Value_A_nl2 = Value_A_nl * TFP_change

Value_added.NL = Value_A_nl2.values



#%%
new_final = x - Value_added
new_final = pd.DataFrame(new_final)

input_output_matrix = Z.copy()
row_totals = x - Y_agg
column_totals = x - Value_added
max_iterations=20
tolerance=1e-4
#%% Error reductions
input_output_matrix[input_output_matrix == 0] = 0.000000001
row_totals[row_totals == 0] = 0.000000001
column_totals[column_totals == 0] = 0.0000001

# adjusted_matrix = ras_algorithm(input_output_matrix, row_totals, column_totals)
# print("Adjusted Matrix:")
# print(adjusted_matrix)
#%%

#def ras_algorithm(input_o{utput_matrix, row_totals, column_totals,{ max_iterations=1000, tolerance=1e-6):
# Ensure input matrices are numpy arrays
input_output_matrix = np.array(input_output_matrix)
row_totals = np.array(row_totals)
column_totals = np.array(column_totals)

# Get the dimensions of the input-output matrix
num_rows, num_cols = input_output_matrix.shape

# Initialize scaling factors
row_scaling = np.ones(num_rows)
column_scaling = np.ones(num_cols)

# Initialize iteration counter
iteration = 0

# Iterate until convergence or maximum iterations reached
while iteration < max_iterations:
    # Step 1: Row Adjustment
    row_adjusted_matrix = (input_output_matrix.T / (column_scaling + np.finfo(float).eps)).T
    row_adjusted_matrix *= row_scaling[:, np.newaxis]
    #np.matmul(np.diag(row_scaling), row_adjusted_matrix) #np.diag(row_scaling)#
  
    # Step 2: Column Adjustment
    column_totals_adjusted = row_adjusted_matrix.sum(axis=0)
    column_scaling = column_totals / (column_totals_adjusted + np.finfo(float).eps)
    

       # Step 3: Scaling
    row_totals_adjusted = row_adjusted_matrix.sum(axis=1)
    row_scaling = row_totals / (row_totals_adjusted + np.finfo(float).eps)
    
    # Calculate the difference between adjusted and original matrix
    diff = np.abs((row_adjusted_matrix * column_scaling).sum(axis=1) - row_totals)
    
    # Check for convergence
    if np.all(diff < tolerance):
        break
    

# Increment iteration counter
    iteration += 1

# Return the adjusted matrix
final_matrix =  row_adjusted_matrix * column_scaling


#%%
full_output = final_matrix.sum(axis = 0) + Y_agg
full_input = final_matrix.sum(axis = 1) + Value_added


full_output.index = Z.index
full_input.index = Z.index
diff_new = full_input - full_output

full_output.loc["NL"].plot()
full_input.loc["NL"].plot()
#diff_new.loc["NL"].plot()

full_output.loc["NL"].sum()
full_input.loc["NL"].sum()

#%%
# row_sum = final_matrix.sum(axis = 0)
# column_sum = final_matrix.sum(axis = 1)
# row_sum = pd.DataFrame(row_sum)
# row_sum.index = Z.index
# column_sum = pd.DataFrame(column_sum)
# column_sum.index = Z.index
# column_sum.sum()
# row_sum.sum()


# #%%
# full_output.plot()
# full_input.plot()
# #%%
# column_sum.loc["NL"].plot()
# row_sum.loc["NL"].plot()
