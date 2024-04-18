# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:38:03 2024

@author: regin
"""

import mario
from mario import slicer, parse_exiobase_3

# %%
Material = "Steel"  # change to focus on material of interest
Resource = "Iron ores"
regionlock = False #nl and rest of the world
regionlock2 = True # bigger regions in the world
sectorlock = True # arbitary sector combinations
#%%
iot_path = r"C:/Industrial_ecology/Thesis/IOT_2021_ixi"
save_path = r'C:/Industrial_ecology/Thesis/Circularinterventions/Code/MR_HIOT_2011_v3_3_18_by_product_technology.csv'
save_path2 = r"C:/Industrial_ecology/Thesis/Circularinterventions/Data"
world_IOT = parse_exiobase_3(path=iot_path, version='3.8.1')
#world_IOT.get_shock_excel(path=save_path)

world_hiot = parse_exiobase_3(path = save_path2, unit = "Hybrid")
