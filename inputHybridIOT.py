# -*- coding: utf-8 -*-
"""
Created on Tue May 28 20:57:08 2024

@author: regin
"""

import pandas as pd
import numpy as np

#%% input path of the IOT 
iot_path = r"C:/Industrial_ecology/Thesis/IOT_2015_ixi"
save_path = r'C:/Industrial_ecology/Thesis/Circularinterventions/Code'
#%%Input of the HIOT 
hybrid_output_path = "C:/Industrial_ecology/Thesis/HIOT_2021_ixi"
F_imp_hh = pd.read_csv(f'{hybrid_output_path}/impacts/F_Y.txt' , sep='\t', index_col=[0], header=[0, 1])

Z_labels = pd.read_csv(f'{iot_path}/Z.txt', sep='\t', index_col=[0, 1], header=[0, 1])
A = pd.read_csv(f'{iot_path}/A.txt', sep='\t', index_col=[0, 1], header=[0,1])
Y = pd.read_csv(f'{iot_path}/Y.txt' , sep='\t', index_col=[0, 1], header=[0, 1])

hiot_path = "C:/Industrial_ecology/Thesis/Circularinterventions/Data/"
Z_hybrid = pd.read_csv(f"{hiot_path}MR_HIOT_2011_v3_3_18_by_product_technology.csv", index_col=[0,1,2,3,4], header=[0,1,2,3])
Z_hybrid = pd.DataFrame(Z_hybrid.values, columns = Z_hybrid.columns, index = Z_hybrid.columns)
Z_hybrid = Z_hybrid.droplevel([2,3], axis=1).droplevel([2,3], axis=0) 
Y_hybrid = pd.read_csv(f"{hiot_path}MR_HIOT_2011_v3_3_18_FD.csv", index_col=[0,1,2,3,4], header=[0,1,2,3])
Y_hybrid = pd.DataFrame(Y_hybrid.values, columns = Y_hybrid.columns, index = Z_hybrid.columns)
Y_hybrid = Y_hybrid.droplevel([2,3], axis=1) 
x_out = Z_hybrid.sum(axis = 1) + Y_hybrid.sum(axis = 1)
x_out[x_out!=0] = 1/x_out[x_out!=0]

A_hybrid = Z_hybrid @ np.diag(x_out)
A_hybrid = pd.DataFrame(A_hybrid.values, columns = Z_hybrid.columns, index = Z_hybrid.columns)

hybrid_output_path = "C:/Industrial_ecology/Thesis/HIOT_2021_ixi"

A_hybrid.to_csv(f'{hybrid_output_path}/A.txt', sep='\t', index=True)  
Y_hybrid.to_csv(f'{hybrid_output_path}/Y.txt',sep='\t', index=True)
Y_hybrid.to_csv(f'{hybrid_output_path}/satellite/F_Y.txt',sep='\t', index=True)
Y_hybrid.to_csv(f'{hybrid_output_path}/satellite/F_Y.txt',sep='\t', index=True)
Y_hybrid.to_csv(f'{hybrid_output_path}/impacts/F_Y.txt',sep='\t', index=True)

world_IOT = parse_exiobase_3(path = hybrid_output_path)


#%%
file_path = 'C:/Industrial_ecology/Thesis/Circularinterventions/Code/Input_circular_interventions/shocks_full.xlsx'
# Specify the sheet name you want to read
sheet_name = 'z'  # Replace with the name of your sheet
# Read the specified sheet into a DataFrame
df = pd.read_excel(file_path, sheet_name=sheet_name)
# Display the DataFrame
print(df)


#%%
A_modify = A_hybrid.copy()
for _, row in df.iterrows():
    country_row = row['row region']
    sector_row = row['row sector']
    country_column = row['row region']
    sector_column = row['column sector']
    value = row['value']
    
    #print(value, category_1_column)
    # Perform the operation using the row and column indices
    #A_hybrid.loc[pd.IndexSlice[:, category_1_row], pd.IndexSlice[:, category_1_column]] *= 1 + value
    A_modify.loc[(country_row, sector_row), (country_column, sector_column)] *= 1 + value

# # Display the updated A_hybrid
# print(A_hybrid.loc[pd.IndexSlice[:, "Re-processing of secondary aluminium into new aluminium"], pd.IndexSlice[:, "AT"]])

# value = A_hybrid.loc[('NL', 'Re-processing of secondary aluminium into new aluminium'), ('NL', 'Re-processing of secondary aluminium into new aluminium')]

A_modify = A_modify.sum(axis = 1)
sortedHybrid = A_modify.groupby(level=0, axis=0, sort=False).sum()

A_hybrid = A_hybrid.sum(axis = 1)
sortedHybrid2 = A_hybrid.groupby(level=0, axis=0, sort=False).sum()


diffchecker = pd.DataFrame()
diffchecker["baseline"] = sortedHybrid2
diffchecker["changes"] = sortedHybrid