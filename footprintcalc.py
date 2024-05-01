# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:15:16 2024

@author: regin
"""
# Import modules
import pandas as pd
import numpy as np
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


#%%
I = np.identity(A.shape[0])
L = np.linalg.inv(I-A)
x = L @ Y.sum(axis=1)
#%%
x_inv = x.copy()
x_inv[x_inv!=0] = 1/x_inv[x_inv!=0]
#%%
indicator = "Domestic Extraction Used - Metal Ores - Bauxite and aluminium ores"
indexfsat = F_sat.index.sort_values()
#%%
F_energy_use = F_sat.loc[indicator]
F_hh_energy_use = F_sat_hh.loc[indicator]
#%%
# Intensities 
f_energy_use = F_energy_use * x_inv
Y_reg = Y.groupby(level=0, axis=1, sort=False).sum()
e_baseline = f_energy_use @ L @ Y_reg + F_hh_energy_use.groupby(level=0, axis=0, sort=False).sum()
e_baseline.sort_values().iloc[[0, -1]]

#%%
Lnew = np.linalg.inv(I-Anew)
xnew = Lnew @ Y.sum(axis = 1)
xnew_inv = xnew.copy()
xnew_inv[xnew_inv!=0] = 1/xnew_inv[xnew_inv!=0]
f_scene = F_energy_use * xnew_inv
Ynew_reg = Ynew.groupby(level=0, axis=1, sort=False).sum()


#%%
e_scene = f_scene @ Lnew @ Ynew_reg + F_hh_energy_use.groupby(level=0, axis=0, sort=False).sum()
print(e_scene.sort_values().iloc[[0, -1]])
print(e_baseline.sort_values().iloc[[0, -1]])

changes = e_scene - e_baseline
print(changes)

frame = {"baseline": e_baseline,
         "scenario": e_scene,
         "difference": changes}

df = pd.DataFrame(frame)
#%%
df.loc["NL"].plot(kind = "bar", title = indicator)
df.plot(kind = "bar", title = indicator, rot=0, figsize=(15,10), fontsize=12)


df.to_excel(f"{outputpath}output.xlsx")  

#%%Production output E = diag(f)x



