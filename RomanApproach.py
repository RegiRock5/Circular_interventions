# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:27:39 2024

@author: regin
"""

#import pandas as pd
import numpy as np
import pandas as pd

#%% Calcualte the Total factor productivity 
TFP_2010 = 0.54
TFP_growthrate = 1.055 
TFP_2050 = TFP_2010 * (TFP_growthrate**8)
TFP_change = TFP_2050/TFP_2010
Z_change = (1- TFP_2050) / (1 - TFP_2010) 
current_pop = 17982825
project_pop = 17982825* 2 #20610000
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
x = L @ Y.sum(axis=1)

#%%  prepare necessary data
Z = A @ (np.diag(x))
Z.columns = Z.index
Va = x - Z.sum(axis=1)

#%%Value added
Y_agg = Y.groupby(level=0, axis=1, sort=False).sum()
xdf = pd.DataFrame(x, index= Z.index, columns=["output orginal"])

#%% Apply changes in Final demand, Value added and Z 

Va_nl = Va.NL
Va_nl = Va_nl * TFP_change

Z_nl = Z.loc[:,"NL"]
Z_nl = Z_nl * TFP_change

Y_agg_nl = Y_agg.NL
Y_agg_nl = Y_agg_nl * TFP_change

#%%apply changes back into the full dataset
Va.NL = Va_nl.values
Z.loc[:,"NL"] = Z_nl
Y_agg.NL = Y_agg_nl
#%% Changed Input output 

x_new = Z.sum(axis=1) + Va 
xdf["new input"] = x_new

print(xdf.loc["NL"])
xdfnl = xdf.loc["NL"]
print(xdf.sum(axis = 0))


new_final = x - Va
new_final = pd.DataFrame(new_final)
xdfnl.plot()

#%%final check 
#output

#input
new_input = Z.sum(axis = 1) + Va
new_output = Z.sum(axis = 0) + Y_agg.sum(axis = 1)

resultsdf = pd.DataFrame()
resultsdf["output"] = new_output
resultsdf["input"] = new_input

print(resultsdf.sum(axis = 0))
resultsdf.plot()


print(new_input.sort_values())
print(new_output.sort_values())
print(xdf.sort_values(by = "output orginal"))


#%% population growth is done on the Leontief inverse
new_output_ = new_output.copy()
new_output_[new_output_!=0] = 1/new_output_[new_output_!=0]
A_2015_no_growth =  Z @ np.diag(new_output_)
I = np.identity(A_2015_no_growth.shape[0])
L_2015_no_growth = np.linalg.inv(I - A_2015_no_growth)

print(Y.NL)
# Modify the final demand of the Netherlands to match a new population
Y.NL = Y.NL * Populationgrowth
print(Y.NL)
Y_agg = Y.groupby(level=0, axis=1, sort=False).sum()
x = L @ Y.sum(axis=1)
Z_popgrowth = A @ (np.diag(x))

newnewoutput = Z.sum(axis = 0) + Y.sum(axis = 1)

#%%

resultsdf["popgrowth"] = newnewoutput
resultsdf.loc["NL"].plot()

print(resultsdf.sum(axis = 0))

print(resultsdf.loc["NL"].sum(axis = 0))
