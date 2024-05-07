# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:05:31 2024

@author: regin
"""

 #Import modules
import pandas as pd
import numpy as np
from multiprocessing import Pool

#%% Calcualte the Total factor productivity 
TFP_2010 = 0.54
TFP_growthrate = 1.055 
TFP_2050 = TFP_2010 * (TFP_growthrate**8)
TFP_change = TFP_2050/TFP_2010
Z_change = (1- TFP_2050) / (1 - TFP_2010) 
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
#%%
# Import impact accounts
F_imp = pd.read_csv(f'{path}impacts/F.txt' , sep='\t', index_col=[0], header=[0, 1])
F_imp_hh = pd.read_csv(f'{path}impacts/F_hh.txt' , sep='\t', index_col=[0], header=[0, 1])

#%% Make the Z matrix using the A and Y
I = np.identity(A.shape[0])
L = np.linalg.inv(I - A)
x = L @ Y.sum(axis=1)

#%%
Z = A @ (np.diag(x))
Z.columns = Z.index

#%%Value added
indicator = "Value Added" 
Y_reg = Y.groupby(level=0, axis=1, sort=False).sum()
xdf = pd.DataFrame(x, index= Z.index)

#%%
#Value_added.NL

#%%
Value_added = x - Z.sum(axis=1)
#print(1 - (Value_added.sum().sum() / Z.sum().sum()))
Value_A_nl = Value_added.NL
Value_A_nl2 = Value_A_nl * TFP_change
#print(1 - (Value_A_nl2.sum().sum() / x.sum()))

Z_nl = Z.loc[:,"NL"]
Z_nl = Z_nl / TFP_change
Z_new = Z.copy()
#Znew.loc[:, "NL"] = Z_nl.values

print(1 - (Value_A_nl.sum().sum() / x.sum()))



Value_added.NL = Value_A_nl2.values



#%%
x_new = Z.sum(axis=1) + Value_added 
xdf["new"] = x_new

print(xdf.loc["NL"])
xdfnl = xdf.loc["NL"]
print(xdf.sum(axis = 0))


new_final = x - Value_added
new_final = pd.DataFrame(new_final)

#%%
# #%%
# goal =  new_final.copy()#.reset_index()
# #goal = Z_new.copy()
# seed = Z_new.copy()#.reset_index()
#     # goal.drop(columns=goal.columns[[0,1]], axis=1, inplace=True)
#     #seed.drop(columns=seed.columns[[0]], axis=1, inplace=True)
    
    
# # Initialise
# goal_get_total = np.array(goal.iloc[:, -1])
# goal_getc_total = np.array(goal.iloc[-1, :])
# goal_row_totals = np.array(goal)
# goal_col_totals = np.array(goal.T)
# matrix = np.array(seed)



# # Replace zeros and blank so division does not create errors
# matrix = np.nan_to_num(matrix)
# goal_col_totals = np.nan_to_num(goal_col_totals)
# goal_row_totals = np.nan_to_num(goal_row_totals)
# matrix[matrix == 0] = 0.0000000001
# goal_col_totals[goal_col_totals == 0] = 0.0000000001
# goal_row_totals[goal_row_totals == 0] = 0.0000000001

# # Iterate until total GDP is within tolerance
# count = 1
# while abs(count) > 0.99:
#     # Row update
#     row_scalars = matrix.sum(axis=1)/goal_row_totals
#     matrix = (matrix.T/row_scalars).T

#     # Column update
#     col_scalars = matrix.sum(axis=0)/goal_col_totals
#     matrix = (matrix/col_scalars)
#     count = np.sum(matrix)-np.sum(goal_col_totals)

# global goal_new
# goal_new = pd.DataFrame(matrix)
# goal_new.to_excel('RAS.xlsx', 'goal')


    
#%%
goal = new_final.copy()  # Assuming new_final is defined elsewhere
seed = Z_new.copy()

# Replace zeros and blanks so division does not create errors
seed[seed == 0] = 0.0000000001
goal[goal == 0] = 0.0000000001


# Convert to numpy arrays
goal_row_totals = np.array(seed.sum(axis=1)) # volgens mij moet dit de orgineel zijn
goal_col_totals = np.array(goal.sum(axis=1)) # deze klopt omdat dit de gereduceerde x is
matrix = np.array(seed)

# Initialise
count = np.inf  # Initialize count to a large value

# Iterate until convergence or max iterations
max_iterations = 1000
tolerance = 0.0000000000000000001
iteration = 0

#%%
while count > tolerance and iteration < max_iterations:
    # Row update
    row_scalars = matrix.sum(axis=1) / goal_row_totals
    matrix = (matrix.T / row_scalars).T

    # Column update
    col_scalars = matrix.sum(axis=0) / goal_col_totals
    matrix = matrix / col_scalars

    count = np.abs(np.sum(matrix) - np.sum(goal_col_totals))
    iteration += 1

# Convert result back to DataFrame
goal_new = pd.DataFrame(matrix)
# file_name = "C:/Industrial_ecology/Thesis/Circularinterventions/Code/Output/RAS.xlsx"
# writer = pd.ExcelWriter(file_name, engine="xlsxwriter")

# goal_new.to_excel(writer, index=False, header = True)
# Save to Excel
#goal_new.to_excel('RAS.xlsx', 'goal')

goal_new.to_csv("C:/Industrial_ecology/Thesis/Circularinterventions/Code/Output/RAS.csv", index=False)
#%%
print(goal_new.sum(axis = 0))
print (goal)

#%%
goal.index = goal.index.T
goal.index = new_final.index

goal_new.index = new_final.index

print(goal_new.loc["NL"].sum(1))
print (goal.loc["NL"])
#%%
goal_new.loc["NL"].sum(1).plot()
goal.loc["NL"].plot()

#%%
goalnl = goal.loc["NL"]
goalnewnl = goal_new.loc["NL"].sum(1)#Added
goalnl.sort_values(ascending=False).iloc[:5]

np.sort_values(goalnl)

print(goal.sum().sum())
print(goal_new.sum().sum())

newdf = pd.DataFrame()
newdf["org"] = goalnl
newdf["new"] = goalnewnl

#%%
# import pandas as pd

# output_df = pd.DataFrame({'Data': [10, 20, 30, 20, 15, 30, 45]})

# file_name ='pandas_simple.xlsx'

# writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
# output_df.round(2)
# output_df.to_excel(writer, index = False, header=True )

# writer.book.use_zip64()

# writer.save()
#%%

# def update_row(matrix_row, goal_row_total):
#     return matrix_row / goal_row_total

# def update_col(matrix_col, goal_col_total):
#     return matrix_col / goal_col_total

# goal = new_final.copy()  # Assuming new_final is defined elsewhere
# seed = Z_new.copy()

# # Replace zeros and blanks so division does not create errors
# seed[seed == 0] = 0.0000000001
# goal[goal == 0] = 0.0000000001

# #%%
# # Convert to numpy arrays
# goal_row_totals = np.array(goal)
# goal_col_totals = np.array(goal)
# matrix = np.array(seed)

# # Initialise
# count = np.inf  # Initialize count to a large value

# # Iterate until convergence or max iterations
# max_iterations = 1
# tolerance = 0.001
# iteration = 0

# #%%
# while count > tolerance and iteration < max_iterations:
#     # Row update
#     with Pool() as pool:
#         updated_rows = pool.starmap(update_row, zip(matrix, goal_row_totals))
#     matrix = np.array(updated_rows)

#     # Column update
#     with Pool() as pool:
#         updated_cols = pool.starmap(update_col, zip(matrix.T, goal_col_totals))
#     matrix = np.array(updated_cols).T

#     count = np.abs(np.sum(matrix) - np.sum(goal_col_totals))
#     iteration += 1

# # Convert result back to DataFrame
# goal_new = pd.DataFrame(matrix)

# # Save to Excel
# goal_new.to_excel('RAS.xlsx', 'goal')


#%%
# import numpy as np
# from numba import jit

# @jit(nopython=True)
# def update_matrix(matrix, goal_row_totals, goal_col_totals):
#     rows, cols = matrix.shape
#     for i in range(rows):
#         row_total = 0
#         for j in range(cols):
#             matrix[i, j] /= goal_row_totals[i]
#             row_total += matrix[i, j]
#         for j in range(cols):
#             matrix[i, j] *= goal_col_totals[j] / row_total
#     return matrix

# goal = new_final.copy()  # Assuming new_final is defined elsewhere
# seed = Z_new.copy()

# # Replace zeros and blanks so division does not create errors
# seed[seed == 0] = 0.0000000001
# goal[goal == 0] = 0.0000000001


# # Convert to numpy arrays
# goal_row_totals = np.array(goal.sum(axis=1))
# goal_col_totals = np.array(goal.sum(axis=0))
# matrix = np.array(seed)

# # Initialise
# count = np.inf  # Initialize count to a large value

# # Iterate until convergence or max iterations
# max_iterations = 2
# tolerance = 0.01
# iteration = 0

# while count > tolerance and iteration < max_iterations:
#     matrix = update_matrix(matrix, goal_row_totals, goal_col_totals)
#     count = np.abs(np.sum(matrix) - np.sum(goal_col_totals))
#     iteration += 1

# # Save result
# np.savetxt('RAS_result.csv', matrix, delimiter=',')



#%%

import numpy as np 

def ras_algorithm(input_output_matrix, row_totals, column_totals, max_iterations=1000, tolerance=1e-6):
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
        row_adjusted_matrix = (input_output_matrix.T / column_scaling).T
        row_adjusted_matrix *= row_scaling[:, np.newaxis]
        
        # Step 2: Column Adjustment
        column_totals_adjusted = row_adjusted_matrix.sum(axis=0)
        column_scaling = column_totals / column_totals_adjusted
        
        # Step 3: Scaling
        row_totals_adjusted = row_adjusted_matrix.sum(axis=1)
        row_scaling = row_totals / row_totals_adjusted
        
        # Calculate the difference between adjusted and original matrix
        diff = np.abs((row_adjusted_matrix * column_scaling).sum(axis=1) - row_totals)
        
        # Check for convergence
        if np.all(diff < tolerance):
            break
        
        # Increment iteration counter
        iteration += 1
    
    # Return the adjusted matrix
    return row_adjusted_matrix * column_scaling

# Example usage:
input_output_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
row_totals = [10, 20, 30]
column_totals = [12, 15, 18]

adjusted_matrix = ras_algorithm(input_output_matrix, row_totals, column_totals)
print("Adjusted Matrix:")
print(adjusted_matrix)