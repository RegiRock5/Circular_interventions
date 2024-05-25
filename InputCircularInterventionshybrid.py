# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:38:03 2024

@author: regin
"""

import mario
from mario import slicer, parse_exiobase_3
import pandas as pd

# %%
Material = "Steel"  # change to focus on material of interest
Resource = "Iron ores"
Material2 = "Aluminium"
Resource2 = "Bauxite and aluminium ores"
regionlock = False #nl and rest of the world
regionlock2 = True # bigger regions in the world
sectorlock = True # arbitary sector combinations
# #%%
# #hiot_path = r"C:/Industrial_ecology/Thesis/IOT_2021_ixi"
# hiot_path = r'C:/Industrial_ecology/Thesis/Circularinterventions/Data/data_hiot'
# save_path = r"C:/Industrial_ecology/Thesis/Circularinterventions/Data"
# #world_HIOT = parse_exiobase_3(path=iot_path, version='3.8.1')
# #world_IOT.get_shock_excel(path=save_path)

# world_hiot = mario.hybrid_sut_exiobase(path = hiot_path)
# save_path = r'HIOTshock_iot.xlsx'
# world_hiot.get_shock_excel(path=save_path)


#%%
extensions = pd.ExcelFile("C:/Industrial_ecology/Thesis/Circularinterventions/Data/MR_HIOT_2011_v3_3_18_extensions.xlsx""")
extensions.sheet_names
# Resource extraction matrix of Z
RE = extensions.parse(sheet_name="resource_act", index_col=[0,1], header=[0,1,2,3]) 
# Resource extraction matrix of Y
RE_FD = extensions.parse(sheet_name="resource_FD", index_col=[0,1], header=[0,1,2,3]) 
# Waste supply matrix of Z
WS = extensions.parse(sheet_name="waste_sup_act", index_col=[0,1], header=[0,1,2,3]) 
# Waste supply matrix of Y
WS_FD = extensions.parse(sheet_name="waste_sup_FD", index_col=[0,1], header=[0,1,2,3]) 

# Waste use matrix of Z
WU = extensions.parse(sheet_name="waste_use_act", index_col=[0,1], header=[0,1,2,3]) 
# Waste use matrix of Y => This is all 0's so it can also be ignored
WU_FD = extensions.parse(sheet_name="waste_use_FD", index_col=[0,1], header=[0,1,2,3]) 

#%%
# Stock addition matrix of Z
SA = extensions.parse(sheet_name="stock_addition_act", index_col=[0,1], header=[0,1,2,3]) 
# Stock addition matrix of Y
SA_FD = extensions.parse(sheet_name="stock_addition_fd", index_col=[0,1], header=[0,1,2,3]) 


#%%
# Stock depletion matrix
SD = extensions.parse(sheet_name="waste_from_stocks", index_col=[0,1], header=[0,1,2,3])


#%%
EM = extensions.parse(sheet_name="Emiss_act", index_col=[0,1,2], header=[0,1,2,3])
EM_FD = extensions.parse(sheet_name="Emiss_FD", index_col=[0,1,2], header=[0,1,2,3])


print(WU.NL.loc[Material])
print(RE.loc[Resource])
