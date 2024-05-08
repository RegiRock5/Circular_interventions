# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:48:56 2024

@author: regin
"""

import numpy as np
import pandas as pd
#%%
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
# input_output_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# row_totals = [8, 17, 26]
# column_totals = [11, 14, 13]
# gross_output = [15, 20, 30]
# max_iterations=200

# tolerance=0.005
# # Ensure input matrices are numpy arrays
# input_output_matrix = np.array(input_output_matrix)
# row_totals = np.array(row_totals)
# column_totals = np.array(column_totals)

# # Get the dimensions of the input-output matrix
# num_rows, num_cols = input_output_matrix.shape

# # Initialize scaling factors
# row_scaling = np.ones(num_rows)
# column_scaling = np.ones(num_cols)

# # Initialize iteration counter
# iteration = 0

# # Iterate until convergence or maximum iterations reached
# while iteration < max_iterations:
#     # Step 1: Row Adjustment
#     row_adjusted_matrix = (input_output_matrix.T / (column_scaling + np.finfo(float).eps)).T
#     row_adjusted_matrix *= row_scaling[:, np.newaxis]
    
#     # Step 2: Column Adjustment
#     column_totals_adjusted = row_adjusted_matrix.sum(axis=0)
#     column_scaling = column_totals / (column_totals_adjusted + np.finfo(float).eps)
    
#     # Step 3: Scaling
#     row_totals_adjusted = row_adjusted_matrix.sum(axis=1)
#     row_scaling = row_totals / (row_totals_adjusted + np.finfo(float).eps)
    
    
#     # Calculate the difference between adjusted and original matrix
#     diff = np.abs((row_adjusted_matrix * column_scaling[:, np.newaxis]).sum(axis=1) - row_totals)
#     iteration += 1
    
#     # Check for convergence
#     if np.all(diff < tolerance):
#         break
    
#     # Increment iteration counter

# # Return the adjusted matrix
# final_matrix =  row_adjusted_matrix * column_scaling

# final_matrix2 = (np.diag(column_scaling)) @ final_matrix @ (np.diag(row_scaling))

# # Example usage:
# #print(final_matrix)
# print(final_matrix.sum(axis = 1))
# print(final_matrix.sum(axis = 0))


#%%
# input_output_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# row_totals = [8, 17, 26]
# column_totals = [11, 14, 13]
# gross_output = [15, 20, 30]

#%% Data processing
labels = pd.Index(["Agriculture", "Manufacturing", "Services"])
input_output_matrix = pd.DataFrame([
    [0.6,2.6,0.5],
    [0.8, 30.6, 7.8],
    [0.9,12.1,23]], index=labels, columns=labels)

Y = np.array([1.9, 28.5, 47.8])
V = np.array([3.30, 22.4, 52.5])
Vstar = V *  1.05**8 
Ystar = Y * 1.05**8


Z_sum = input_output_matrix.sum(axis=1) 
x_out = Z_sum + Y
x_in = input_output_matrix.sum(axis=0) + V 
x_innew = input_output_matrix.sum(axis=0) + Vstar

#Row totals should be equal to sum of z rows
row_totals = x_out - Ystar
#The changed column totals reflect the sum of columns with total input (sum of z + V - the changed value added)
column_totals = x_in - Vstar
#%% Perform the RAS algorithm
iteration = 0
max_iterations=1000
tolerance=0.05

input_output_matrix = np.array(input_output_matrix)
row_totals = np.array(row_totals)
column_totals = np.array(column_totals)
x_innew = np.array(x_innew)

A = input_output_matrix @ np.linalg.inv(np.diag(x_out)) #original technical coefficient matrix A0

while iteration < max_iterations:
    iteration += 1
    col_scalars = np.diag(column_totals) @ (np.linalg.inv(np.diag(input_output_matrix.sum(axis=0)))) #s1
    A = A @ col_scalars #(np.linalg.inv(np.diag(col_scalars)))
    input_output_matrix = A @ (np.diag(x_innew))
    print("col sums:", col_scalars.sum(axis = 0))
    #print("COl sums:", A)
    # print("Row scalars:", row_scalars.sum(axis = 1))
    # print("Column scalars:", col_scalars.sum(axis = 1))
    diff = np.abs((input_output_matrix.sum(axis=1)) - row_totals)
    if np.all(diff < tolerance):
        print("Convergence achieved after in rows", iteration, "iterations.")
        break

    
    print("Iteration:", iteration)
    # A = input_output_matrix @ np.linalg.inv(np.diag(x_innew))
    input_output_matrix = A @ (np.diag(x_innew)) #Z1
    row_scalars = np.diag(row_totals) @ (np.linalg.inv(np.diag(input_output_matrix.sum(axis=1)))) #r1
    A = A @ row_scalars#(np.linalg.inv(np.diag(row_scalars)))   #A(1)
    input_output_matrix = A @ (np.diag(x_innew))        
    print("Row sums:", input_output_matrix.sum(axis = 1))
    #print("Row sums:", A)
    
    diff = np.abs((input_output_matrix.sum(axis=0)) - column_totals)
    # Check for convergence
    if np.all(diff < tolerance):
        print("Convergence achieved after columns", iteration, "iterations.")
        break
    
    
else:
    print("Maximum iterations reached without convergence.")

    
# final_matrix2 = col_scalars @ A @ row_scalars
# final_matrix2 = A @ (np.diag(x_innew))


print(input_output_matrix)
print(input_output_matrix.sum(axis = 1)) # rows
print(row_totals)
print(input_output_matrix.sum(axis = 0)) #columns
print(column_totals)

#%%
# print(final_matrix2)
# print(final_matrix2.sum(axis = 1))
# print(final_matrix2.sum(axis = 0))

#%%Calculate the gross output of the gross output and input 
full_output = input_output_matrix.sum(axis = 1) + Ystar
full_input = input_output_matrix.sum(axis = 0) + Vstar
delta_output = input_output_matrix.sum(axis = 1)
delta_input = input_output_matrix.sum(axis = 0)
#%%
resultsdf = pd.DataFrame()
# resultsdf["output"] = full_output
# resultsdf["input"] = full_input
resultsdf["base output"] = delta_output
resultsdf["base input"] = delta_input
resultsdf["original output"] = input_output_matrix.sum(axis=1) 
resultsdf["original input"] = input_output_matrix.sum(axis=0) 


resultsdf.plot(kind= 'bar') # quick check of the difference between the output - input (should be equal)
