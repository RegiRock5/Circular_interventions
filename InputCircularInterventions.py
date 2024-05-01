import mario
from mario import slicer, parse_exiobase_3

# %%
regionlock = False #nl and rest of the world
regionlock2 = False # bigger regions in the world
sectorlock = False # arbitary sector combinations
outputpath = "C:/Industrial_ecology/Thesis/Circularinterventions/outputdata/" 

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
    
    
#%%

#world_IOT.get_aggregation_excel(path=r'sector.xlsx', levels= "Sector")
#%%

world_IOT.shock_calc(io=r'shock_iot.xlsx', # Path to the excel file
                z= True, # the shock will be implemented on z
                notes=['you may add some notes for metadata']
              )

#new scenario is added to the list of scenarios
print(world_IOT.scenarios)

#the shock is recorded on metadata
#world_IOT.meta_history

#lets have a look on the changes
world_IOT['shock 1']['Y']-world_IOT['baseline']['Y']

#%%
world_IOT.shock_calc(io=r'shock_iot2.xlsx', # Path to the excel file
                z= True, # the shock will be implemented on z
                Y = True,
                notes=['interindustry and final demand changes']
              )
#%%
new_units= world_IOT.units['Satellite account']
print(new_units)
#



#%%CO2 - combustion - air
# world_IOT.plot_matrix(matrix = "X", 
#                       x = "Region_to",
#                       color = "Sector_to",
#                       #filters_Satellite_account = ["CO2 - combustion - air"],
#                       base_scenario='baseline', # printing the delta_x with respect to baseline scenario,
#                       path = "Co2test.html")

#%%
if sectorlock == True: 
    world_IOT.aggregate(r'sector.xlsx', ignore_nan= True, levels = "Sector")
    world_IOT.get_index("Sector")

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


#%%
print(world_IOT['shock 1']['Z'])

New_Z = world_IOT['shock 1']['Z']

#%%
print(world_IOT['shock 1']['X'])

#%%
New_A = world_IOT['shock 1']['z']
New_A2 = world_IOT['shock 2']['z']
New_Y = world_IOT['shock 2']["Y"]

#%%
New_A.to_csv(f'{outputpath}newA.csv', index=False)  
New_A2.to_csv(f'{outputpath}NewA2.csv', index = False) 
New_Y.to_csv(f'{outputpath}NewY.csv', index =False)