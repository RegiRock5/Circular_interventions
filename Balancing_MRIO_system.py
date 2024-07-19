# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:34:21 2024

@author: regin
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% import extensions and slice the data you want to use
hiot_path = "C:/Industrial_ecology/Thesis/Circularinterventions/Data/"
extensions = pd.ExcelFile(f"{hiot_path}MR_HIOT_2011_v3_3_18_extensions.xlsx")
extensions.sheet_names

resource = "Iron ores"
#resource = "Bauxite and aluminium ores"
Emission = "Carbon dioxide, fossil"
#resource = "Copper ores"

# #resource extraction --> take only the material of interest
# RE = extensions.parse(sheet_name="resource_act", index_col=[0,1], header=[0,1,2,3]) 
# RE_FD = extensions.parse(sheet_name="resource_FD", index_col=[0,1], header=[0,1,2,3]) 
# RE = RE.loc[resource].sum(axis = 0)
# RE_FD = RE_FD.loc[resource]
#emissions 
EM = extensions.parse(sheet_name="Emiss_act", index_col=[0,1,2], header=[0,1,2,3])
EM_FD = extensions.parse(sheet_name="Emiss_FD", index_col=[0,1,2], header=[0,1,2,3])

EM = EM.loc[Emission].sum(axis = 0)
EM_FD = EM_FD.loc[Emission]
SUM_CO2_hybrid = (EM.sum().sum() + EM_FD.sum().sum()) / 1000

#/1000
EM = extensions.parse(sheet_name="Emiss_act", index_col=[0,1,2], header=[0,1,2,3])
EM_FD = extensions.parse(sheet_name="Emiss_FD", index_col=[0,1,2], header=[0,1,2,3])
FD_drivers_cat = [
    "Carbon dioxide, fossil",
    "N2O",
    "CH4"]

EM2 = EM.loc[FD_drivers_cat].sum(1)
EM_FD2 = EM_FD.loc[FD_drivers_cat].sum(1)
EM4 = EM2 + EM_FD2
    
EM3 = pd.DataFrame()
EM3["Carbon dioxide, fossil"] = EM4["Carbon dioxide, fossil"]*1
EM3["N2O"] = EM4["N2O"] * 25
EM3["CH4"] = EM4["CH4"] * 298

SUM_GHG_hybrid = EM3.sum().sum() /1000
#%%%
hiot_path = "C:/Industrial_ecology/Thesis/IOT_2011_ixi/"

indicator ="CO2 - combustion - air"
F_sat = pd.read_csv(f'{hiot_path}satellite/F.txt' , sep='\t', index_col=[0], header=[0, 1])
F_sat_hh = pd.read_csv(f'{hiot_path}satellite/F_hh.txt' , sep='\t', index_col=[0], header=[0, 1])
F_indicator = F_sat.loc[indicator]
F_indicator2 = F_sat_hh.loc[indicator]
SUM_sat_CO2 = (F_indicator.sum().sum() + F_indicator2.sum().sum())/1000000

indicator = "Carbon dioxide (CO2) Fuel combustion"
F_imp = pd.read_csv(f'{hiot_path}impacts/F.txt' , sep='\t', index_col=[0], header=[0, 1])
F_imp_hh = pd.read_csv(f'{hiot_path}impacts/F_hh.txt' , sep='\t', index_col=[0], header=[0, 1])
F_indicator = F_imp.loc[indicator]
F_indicator2 = F_imp_hh.loc[indicator]
SUM_impact_CO2 = F_indicator.sum().sum() + F_indicator2.sum().sum()

indicator = "GHG emissions (GWP100) | Problem oriented approach: baseline (CML, 2001) | GWP100 (IPCC, 2007)"
F_imp = pd.read_csv(f'{hiot_path}impacts/F.txt' , sep='\t', index_col=[0], header=[0, 1])
F_imp_hh = pd.read_csv(f'{hiot_path}impacts/F_hh.txt' , sep='\t', index_col=[0], header=[0, 1])
F_indicator = F_imp.loc[indicator]
F_indicator2 = F_imp_hh.loc[indicator]
SUM_GHG_mon = (F_indicator.sum().sum() + F_indicator2.sum().sum())/1000000



#%%

resource = "Iron ores"
resource = "Bauxite and aluminium ores"

#resource extraction --> take only the material of interest
RE = extensions.parse(sheet_name="resource_act", index_col=[0,1], header=[0,1,2,3]) 
RE_FD = extensions.parse(sheet_name="resource_FD", index_col=[0,1], header=[0,1,2,3]) 
RE = RE.loc[resource].sum(axis = 0)
RE_FD = RE_FD.loc[resource]

SUM_AL_hyb = (RE.sum().sum() + RE_FD.sum().sum()) / 1000

#%%
F_sat = pd.read_csv(f'{hiot_path}satellite/F.txt' , sep='\t', index_col=[0], header=[0, 1])
F_sat_hh = pd.read_csv(f'{hiot_path}satellite/F_hh.txt' , sep='\t', index_col=[0], header=[0, 1])
indicator = "Domestic Extraction Used - Metal Ores - Bauxite and aluminium ores"
#indicator ="Domestic Extraction Used - Metal Ores - Iron ores"

F_indicator = F_sat.loc[indicator]
F_indicator2 = F_sat_hh.loc[indicator]
SUM_AL_mon = (F_indicator.sum().sum() + F_indicator2.sum().sum())

#%%
resource = "Iron ores"
RE = extensions.parse(sheet_name="resource_act", index_col=[0,1], header=[0,1,2,3]) 
RE_FD = extensions.parse(sheet_name="resource_FD", index_col=[0,1], header=[0,1,2,3]) 
RE = RE.loc[resource].sum(axis = 0)
RE_FD = RE_FD.loc[resource]

Sum_st_hybrid = (RE.sum().sum() + RE_FD.sum().sum()) / 1000

#%%
F_sat = pd.read_csv(f'{hiot_path}satellite/F.txt' , sep='\t', index_col=[0], header=[0, 1])
F_sat_hh = pd.read_csv(f'{hiot_path}satellite/F_hh.txt' , sep='\t', index_col=[0], header=[0, 1])
indicator ="Domestic Extraction Used - Metal Ores - Iron ores"

F_indicator = F_sat.loc[indicator]
F_indicator2 = F_sat_hh.loc[indicator]
SUM_st_mon = (F_indicator.sum().sum() + F_indicator2.sum().sum())

#%%
