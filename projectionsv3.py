# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:48:56 2024

@author: regin
"""

import numpy as np
import pandas as pd
#%%

input_output_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
row_totals = [8, 17, 26]
column_totals = [11, 14, 13]
gross_output = [15, 20, 30]


A = input_output_matrix @ np.linalg.inv(np.diag(gross_output))

# #%%
# input_output_matrix = np.array(input_output_matrix)
# row_totals = np.array(row_totals)
# column_totals = np.array(column_totals)

# num_rows, num_cols = input_output_matrix.shape

# # Initialize scaling factors
# row_scaling = np.ones(num_rows)
# column_scaling = np.ones(num_cols)

# iteration = 0 

# while iteration < max_iterations:
#     Z = A @ (np.diag(gross_output))
#     TotalRows = Z.sum(axis = 1)
#     row_adjusted = (TotalRows / (row_totals + np.finfo(float).eps))
#     Radjuster = np.diag(row_adjusted)
#     A = A @ Radjuster
    
#     #column adjuster
#     Z = A @ (np.diag(gross_output))
#     Totalcolumns = Z.sum(axis = 0)
#     column_adjusted = Totalcolumns/ (column_totals  + np.finfo(float).eps)
#     Cadjuster = np.diag(column_adjusted)
#     A = A @ Cadjuster
    
#     iteration += 1



# resultsZ = A @ (np.diag(gross_output))
    

#%%
input_output_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
row_totals = [8, 17, 26]
column_totals = [11, 14, 13]
gross_output = [15, 20, 30]
max_iterations=200

tolerance=0.005
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
    
    # Step 2: Column Adjustment
    column_totals_adjusted = row_adjusted_matrix.sum(axis=0)
    column_scaling = column_totals / (column_totals_adjusted + np.finfo(float).eps)
    
    # Step 3: Scaling
    row_totals_adjusted = row_adjusted_matrix.sum(axis=1)
    row_scaling = row_totals / (row_totals_adjusted + np.finfo(float).eps)
    
    
    # Calculate the difference between adjusted and original matrix
    diff = np.abs((row_adjusted_matrix * column_scaling[:, np.newaxis]).sum(axis=1) - row_totals)
    iteration += 1
    
    # Check for convergence
    if np.all(diff < tolerance):
        break
    
    # Increment iteration counter

# Return the adjusted matrix
final_matrix =  row_adjusted_matrix * column_scaling

final_matrix2 = (np.diag(column_scaling)) @ final_matrix @ (np.diag(row_scaling))

# Example usage:
#print(final_matrix)
print(final_matrix.sum(axis = 1))
print(final_matrix.sum(axis = 0))


#%%
input_output_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
row_totals = [8, 17, 26]
column_totals = [11, 14, 13]
gross_output = [15, 20, 30]

max_iterations=200
iteration = 0
max_iterations=1000
tolerance=0.005

input_output_matrix = np.array(input_output_matrix)
row_totals = np.array(row_totals)
column_totals = np.array(column_totals)


while iteration < max_iterations:
    row_scalars = input_output_matrix.sum(axis=1)/row_totals
    input_output_matrix = (input_output_matrix.T/row_scalars).T

    # Column update
    col_scalars = input_output_matrix.sum(axis=0)/column_totals
    input_output_matrix = (input_output_matrix/col_scalars)
    
    # row_scalars = input_output_matrix.sum(axis=1)/row_totals
    # input_output_matrix = (input_output_matrix.T/row_scalars).T
    
    iteration += 1
    
    diff = np.abs((input_output_matrix * row_scalars[:, np.newaxis]).sum(axis=0) - column_totals)
    count = np.sum(input_output_matrix)-np.sum(column_totals)

    # Check for convergence
    if np.all(diff < tolerance):
        break

    if (count) > 0.95:
        break
    
final_matrix2 = (np.diag(col_scalars)) @ input_output_matrix @ (np.diag(row_scalars))



print(input_output_matrix)
print(input_output_matrix.sum(axis = 1))
print(input_output_matrix.sum(axis = 0))

print(final_matrix2)
print(final_matrix2.sum(axis = 1))
print(final_matrix2.sum(axis = 0))

