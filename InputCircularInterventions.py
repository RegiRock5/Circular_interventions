import mario
from mario import slicer, parse_exiobase_3

# %%
regionlock = False #nl and rest of the world
regionlock2 = False # bigger regions in the world
sectorlock = False # arbitary sector combinations
outputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/Code/Output/" 

#%%
iot_path = r"C:/Industrial_ecology/Thesis/IOT_2021_ixi"
save_path = r'C:/Industrial_ecology/Thesis/Circularinterventions/Code'
world_IOT = parse_exiobase_3(path=iot_path, version='3.8.1')
#world_IOT.get_shock_excel(path=save_path)

#%%
if regionlock == True: 
    world_IOT.aggregate(r'Dutch_agg.xlsx', ignore_nan= True, levels = "Region")
    RegionI= "NL"
    world_IOT.get_index("Region")
    
if regionlock2 == True: 
    world_IOT.aggregate(r'Bigregion.xlsx', ignore_nan= True, levels = "Region")
    RegionI= "NL"
    world_IOT.get_index("Region")
    
#%% Uncomment if in need of shockexcel 
#world_IOT.get_aggregation_excel(path=r'sector.xlsx', levels= "Sector")

#%% Implement Aluminiums shocks (shock 1)

world_IOT.shock_calc(io=r'shocks_al.xlsx', # Path to the excel file
                z= True, # the shock will be implemented on z
                Y= True,
                e= True,
                notes=['Implement Aluminium shocks']
              )
#%%Implement Steel shocks

world_IOT.shock_calc(io=r'shocks_st.xlsx', # Path to the excel file
                z= True, # the shock will be implemented on z
                Y = True,
                e = True,
                notes=['Implement Steel shocks']
              )
#%%Implement both aluminium and steel shocks (shock 3)
world_IOT.shock_calc(io=r'shocks_full.xlsx', # Path to the excel file
                z= True, # the shock will be implemented on z
                Y= True,
                e= True,
                notes=['Implement Aluminium shocks']
              )
#%%Implement sectoral aggregation to reduce sector 
if sectorlock == True: 
    world_IOT.aggregate(r'sector.xlsx', ignore_nan= True, levels = "Sector")
    world_IOT.get_index("Sector")
#%%Create variables to perform footprint calculations 
# print(world_IOT['shock 1']['Z'])
# New_Z = world_IOT['shock 1']['Z']

#%%
# print(world_IOT['shock 1']['X'])

#%%
A_Al_adjusted = world_IOT['shock 1']['z']
Y_Al_adjusted = world_IOT['shock 1']['Y']
e_Al_adjusted =world_IOT['shock 1']['E']

A_St_adjusted = world_IOT['shock 2']['z']
Y_St_adjusted = world_IOT['shock 2']['Y']
e_St_adjusted =world_IOT['shock 2']['E']

A_full_adjusted = world_IOT['shock 3']['z']
Y_full_adjusted = world_IOT['shock 3']['Y']
e_full_adjusted =world_IOT['shock 3']['E']

#%%
A_Al_adjusted.to_csv(f'{outputpath}A_Al_adjusted.csv', index=True)  
Y_Al_adjusted.to_csv(f'{outputpath}Y_Al_adjusted.csv', index=True)
e_Al_adjusted.to_csv(f'{outputpath}F_Al_adjusted.csv', index=True)

A_St_adjusted.to_csv(f'{outputpath}A_St_adjusted.csv', index=True)  
Y_St_adjusted.to_csv(f'{outputpath}Y_St_adjusted.csv', index=True)
e_St_adjusted.to_csv(f'{outputpath}F_St_adjusted.csv', index=True)


A_full_adjusted.to_csv(f'{outputpath}A_full_adjusted.csv', index=True)  
Y_full_adjusted.to_csv(f'{outputpath}Y_full_adjusted.csv', index=True)  
e_full_adjusted.to_csv(f'{outputpath}F_full_adjusted.csv', index=True)


#%%
# new_units= world_IOT.units['Satellite account']
# print(new_units)
# #

#%%CO2 - combustion - air
# world_IOT.plot_matrix(matrix = "X", 
#                       x = "Region_to",
#                       color = "Sector_to",
#                       #filters_Satellite_account = ["CO2 - combustion - air"],
#                       base_scenario='baseline', # printing the delta_x with respect to baseline scenario,
#                       path = "Co2test.html")

#%%

#The plot_matrix function can be used to plot the changes of the X
# world_IOT.plot_matrix(
#     matrix='F', # plotting the X matrix
#     x='Region_from', # putting the origin regions on the X axis
#     color='Sector_from', # colors are defined tby the origin sectors
#     base_scenario='baseline', # printing the delta_x with respect to baseline scenario,
#     path = 'delta_X1.html'
#                 )

#%%
# world_IOT.plot_matrix(
#     matrix = 'Y',
#     x = 'Region_from',
#     color = 'Region_to',
#     path= 'final_comnsumtpiton_by_region.html'
#     )
